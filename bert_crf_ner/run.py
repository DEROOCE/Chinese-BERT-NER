import datetime as dt
import torch
import torch.nn as nn
from dataloader import data_loader
from bert_crf import BertCRF
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from opt import get_opts
import mlflow
import mlflow.pytorch


class NER(pl.LightningModule):
    def __init__(self, args, model, steps_per_epoch=None):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.model = model 
        self.steps_per_epoch = steps_per_epoch
        self.args = args 
    
    def forward(self, word_id, mask, label=None):
        return self.model(word_id, mask, label)
    
    def training_step(self, batch, batch_size):
        train_loss = 0
        inputs, labels = batch 
        word_id, attn_mask = inputs["input_ids"], inputs["attention_mask"]
        #print("word_id", word_id)
        attn_mask = torch.gt(attn_mask, 0)
        _, loss = self.forward(word_id, attn_mask, labels)
        
        #loss.backward()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.args.clip_grad)
        self.optimizers().step() 
        #self.scheduler().step()
        self.model.zero_grad()
        train_loss += loss.item()
        mlflow.log_metric("loss", loss)
        self.log("train_loss", train_loss,
                on_epoch=True, prog_bar=True, logger=True)
        print("train_loss = ", train_loss)
        return {"loss": loss}

    
    def test_step(self, batch, batch_size):
        inputs, labels = batch 
        word_id, attn_mask = inputs["input_ids"], inputs["attention_mask"]
        attn_mask = torch.gt(attn_mask, 0)
        _, loss = self.forward(word_id, attn_mask, labels) 
        metrics = {"test_loss": loss}
        return metrics
    

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        model = self.model
        args = self.args
        bert_param_optimizer = list(model.bert.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        liner_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr": args.learning_rate},
            {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": args.learning_rate},

            {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr": args.crf_learning_rate},
            {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": args.crf_learning_rate},

            {"params": [p for n, p in liner_param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, "lr": args.crf_learning_rate},
            {"params": [p for n, p in liner_param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0, "lr": args.crf_learning_rate}
        ]
        # 需要了解 AdamW
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        train_step_per_eopch = self.steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=(args.epoch_num // 10) * train_step_per_eopch,
                                                    num_training_steps=args.epoch_num * train_step_per_eopch)
       
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
        }
        return [optimizer], [scheduler]

def train(args, model, tokenizer):
    cur_time = dt.datetime.now().strftime("%H-%M-%S-%Y-%m-%d")
    ckpt_callback_train_loss = ModelCheckpoint(
                monitor="train_loss", dirpath="./model/pl/", 
                filename='crf-{cur_time}',
                mode="min"
    )
    trainer = pl.Trainer(max_epochs=5, 
                        callbacks=ckpt_callback_train_loss,
                        #accelerator="gpu", devices=[0],
                        log_every_n_steps=5,
                        enable_progress_bar=True
                        )
    mlflow.set_tracking_uri("http://192.168.11.95:5002")
    mlflow.set_experiment("BERT_NER_zh")
    mlflow.start_run(run_name="%s_CRF" 
                        % cur_time)
    train_loader = data_loader(args, tokenizer)
    steps_per_epoch = len(train_loader)
    ner_model = NER(args, model, steps_per_epoch)
    mlflow.pytorch.autolog()
    trainer.fit(ner_model,train_loader)
    mlflow.end_run()
    print("training has been ended")



def evaluate(args, model, tokenizer):
    eval_loader = data_loader(args, tokenizer)
    trainer = pl.Trainer(max_epochs=1,
                        enable_progress_bar=True)
    trainer.test(model, eval_loader, verbose=True )
    print("evaluating has beend ended")

def predict(args, model, tokenizer):
    pred_loader = data_loader(args, tokenizer)
                #输出tag
    ner_tags = {
        "0": "-", # None
        "1": "definition", 
        "2":"experimental_result",
    } 
    for i, batch in enumerate(pred_loader):
        # in predict mode, batch_size = 1
        inputs, labels = batch
        #print(inputs)
        word_id, attn_mask = inputs["input_ids"], inputs["attention_mask"]
        attn_mask = torch.gt(attn_mask, 0)
        #移除pad
        select = inputs['attention_mask'] == 1
        # print("select is :", select)
        input_id = word_id[select]
        # print("out = ", out)
        label = labels[select]
        # print("labels = ", label)
        print("输出原句: ",tokenizer.decode(input_id).replace(' ', ''))
        print("="*100)
        logits = model(word_id, attn_mask)[0]
        #batch_pred_labels_ids = model.model.crf.decode(logits, attn_mask)[:select]
        batch_pred_labels_ids = model.model.crf.decode(logits, attn_mask)[0]
        for m, tag in enumerate([label, batch_pred_labels_ids]):
            s = ""
            for n in range(len(tag)):
                if tag[n] == 0:
                    s += '.'
                    continue 
                
                if tag[n] !=0 and tag[n-1] == 0:
                    ner_tag  = ner_tags[str(tag[n])]
                    s += ner_tag 
                    s += ": "
                s += tokenizer.decode(input_id[n])

            if m == 0:
                print("输出真实标签命名实体：", s)
            else:
                print("输出预测标签命名实体：", s)

        print("="*100)


def main():
    args = get_opts()
    #args.root_dir = './data/train_data1.json'  
    args.do_train = True
    #args.do_eval = True
    #args.do_pred = True
    args.bert_model = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    
    if args.do_train:
        args.root_dir = "./data/train_data.json"
        model = BertCRF.from_pretrained(args.bert_model, num_labels=args.num_class)
        train(args, model, tokenizer)

    if args.do_eval:
        args.model_dir = "./model/pl/epoch=0-step=2.ckpt"
        args.root_dir = "./data/test_data.json"
        args.batch_size = 1
        args.shuffle_data = False
        #model = BertCRF.from_pretrained(args.model_dir)
        model = NER.load_from_checkpoint(args.model_dir)
        evaluate(args, model, tokenizer)

    if args.do_pred:
        args.model_dir = "./model/pl/epoch=0-step=2.ckpt"
        args.root_dir = "./data/test_data.json"
        args.shuffle_data = False
        args.batch_size = 1
        model = NER.load_from_checkpoint(args.model_dir)
        predict(args, model, tokenizer)
        
    
if __name__ == "__main__":
    main()
