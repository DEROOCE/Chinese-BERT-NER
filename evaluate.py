import torch 
from dataloader import data_loader
from transformers import AutoTokenizer  
from transform import reshape_and_remove_pad, get_correct_and_total_count
from opt import get_opts


def test(dataloader, hparams):
    model = torch.load(hparams.model_dir)
    
    model.eval()
    
    correct = 0
    total = 0
    correct_content = 0 
    total_content = 0
    
    for step, (inputs, labels) in enumerate(dataloader):

        with torch.no_grad():
            #[b, lens] -> [b, lens, 3] -> [b, lens]
            outs = model(inputs)

        #对outs和label变形,并且移除pad
        #outs -> [b, lens, 3] -> [c, 3]
        #labels -> [b, lens] -> [c]
        outs, labels = reshape_and_remove_pad(outs, labels,
                                            inputs['attention_mask'],
                                            hparams.num_class)
        print("outs shape:  ", outs.shape)
        # print("labels = ", labels[:200], "label shape:", labels.shape, "\n =========================")
        counts = get_correct_and_total_count(labels, outs, hparams.num_class)
        correct += counts[0]
        total += counts[1]
        correct_content += counts[2]
        total_content += counts[3]

    print("Accuracy: ", correct / total, "Correct content: ", correct_content / total_content)
    

def predict(loader, hparams):
    model = torch.load(hparams.model_dir)
    model.eval()


    for i, (inputs, labels) in enumerate(loader):

        with torch.no_grad():
            #[b, lens] -> [b, lens, 3] -> [b, lens]
            outs = model(inputs).argmax(dim=2)

        print("attention_mask shape:", inputs["attention_mask"].shape)
        # for i in range(batch_size):
        for i in range(5):
            #移除pad
            select = inputs['attention_mask'][i] == 1
            # print("select is :", select)
            input_id = inputs['input_ids'][i, select]
            out = outs[i, select]
            # print("out = ", out)
            label = labels[i, select]
            # print("labels = ", label)
            
            #输出原句子
            print("输出原句: ",tokenizer.decode(input_id).replace(' ', ''))
            print("="*100)    
            #输出tag
            ner_tags = {
                "0": "None",
                "1": "definition", 
                "2":"experimental_result",
            }
            for i, tag in enumerate([label, out]):
                s = ''
                for j in range(len(tag)):
                    if tag[j] == 0:
                        s += '·'
                        continue
                    
                    if tag[j] !=0 and tag[j-1] == 0:    
                        ner_tag = ner_tags[str(tag[j].item())]
                        s += ner_tag
                        s += ": "
                    s += tokenizer.decode(input_id[j])
                    # s += str(tag[j].item())
                if i == 0:
                    print("输出真实标签命名实体部分: ", s)
                else:
                    print("输出预测标签命名实体部分: ", s)                    


            print('='*100)


    
    
    
if __name__ == "__main__":
    hparams = get_opts()  
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # path = "./data/test_data.json"
    # batch_size = 1
    loader = data_loader(hparams, tokenizer)
    test(loader, hparams)
    predict(loader, hparams)
    
    
    