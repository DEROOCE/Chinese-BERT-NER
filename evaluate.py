import torch 
from dataloader import data_loader
from transformers import AutoTokenizer  
from transform import reshape_and_remove_pad, get_correct_and_total_count

def test(dataloader):
    model = torch.load("./model/ner_tune.model")
    
    model.eval()
    
    correct = 0
    total = 0
    correct_content = 0 
    total_content = 0
    
    for step, (inputs, labels) in enumerate(dataloader):

        with torch.no_grad():
            #[b, lens] -> [b, lens, 7] -> [b, lens]
            outs = model(inputs)

        #对outs和label变形,并且移除pad
        #outs -> [b, lens, 7] -> [c, 7]
        #labels -> [b, lens] -> [c]
        outs, labels = reshape_and_remove_pad(outs, labels,
                                            inputs['attention_mask'])
        print("outs shape:  ", outs.shape)
        print("labels = ", labels[:200], "label shape:", labels.shape, "\n =========================")
        counts = get_correct_and_total_count(labels, outs)
        correct += counts[0]
        total += counts[1]
        correct_content += counts[2]
        total_content += counts[3]

    print("Accuracy: ", correct / total, "Correct content: ", correct_content / total_content)
    

def predict(loader):
    model = torch.load('./model/ner_tune.model')
    model.eval()


    for i, (inputs, labels) in enumerate(loader):

        with torch.no_grad():
            #[b, lens] -> [b, lens, 7] -> [b, lens]
            outs = model(inputs).argmax(dim=2)

        print("attention_mask shape:", inputs["attention_mask"].shape)
        for i in range(2):
            #移除pad
            select = inputs['attention_mask'][i] == 1
            input_id = inputs['input_ids'][i, select]
            out = outs[i, select]
            label = labels[i, select]
            
            #输出原句子
            print("输出原句: ",tokenizer.decode(input_id).replace(' ', ''))
            print("="*100)    
            #输出tag
            ner_tags = {
                "1":"background", 
                "2": "definition", 
                "3": "feature", 
                "4": "method", 
                "5":"experimental_result",
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
                if i == 0:
                    print("输出真实标签命名实体部分: ", s)
                else:
                    print("输出预测标签命名实体部分: ", s)                    


            print('='*100)


    
    
    
if __name__ == "__main__":  
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    path = "./data/test_data.json"
    batch_size = 8
    loader = data_loader(path, batch_size, tokenizer)
    predict(loader)
    
    
    
