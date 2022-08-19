import torch
# from transformers import AdamW 
from tqdm import tqdm 
from transformers import AutoModel 
from transformers import AutoTokenizer
from dataloader import data_loader
from model import NER 
from transform import reshape_and_remove_pad, get_correct_and_total_count
from opt import get_opts

def train(model, epochs, loader):
    lr = 2e-5 if model.tuning else 5e-4
    
    # 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    # for epoch in tqdm(range(epochs)):
    for epoch in range(epochs):
        for step, (inputs, labels) in tqdm(enumerate(loader)):
            outs = model(inputs)
            outs, labels = reshape_and_remove_pad(outs, labels,
                                                inputs["attention_mask"],
                                                hparams.num_class)
            
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if step % 10 == 0:
                counts = get_correct_and_total_count(labels, outs)
                acc = counts[0] / counts[1]
                acc_content = counts[2] / counts[3]
                print("epoch=",epoch, " step=", step, " loss=", loss.item(),
                    " accuracy=", acc, " accuracy_content=", acc_content)
        

    torch.save(model, "./model/ner.model")
    
if __name__ == "__main__":
    hparams = get_opts()
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    pretrained = AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")
    # path = "./data/train_data.json"
    # batch_size = 128
    loader = data_loader(hparams, tokenizer)
    epochs = 10
    ner = NER(pretrained, hparams)
    ner.fine_tuning(True)
    # 参数量
    print("模型参数量为",sum(i.numel() for i in ner.parameters()) / 10000, "万.")  # 万为单位
    train(ner, epochs, loader)