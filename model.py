from transformers import AutoModel 
from transformers import AutoTokenizer
from dataloader import data_loader
import torch 

class NER(torch.nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.tuning = False 
        self.pretrained_tune = None 
        self.pretrained = pretrained
        self.rnn = torch.nn.GRU(768, 768, batch_first=True)
        self.fc = torch.nn.Linear(768, 7)

    def forward(self, inputs):
        if self.tuning:
            out = self.pretrained_tune(**inputs).last_hidden_state 
        else:
            with torch.no_grad():
                out = self.pretrained(**inputs).last_hidden_state 
        out, _ = self.rnn(out)
        out = self.fc(out).softmax(dim=2)
        
        return out 
    
    def fine_tuning(self, tuning):
        self.tuning = tuning 
        if tuning:
            for i in self.pretrained.parameters():
                i.requires_grad = True 
                
            self.pretrained.train() 
            self.pretrained_tune = self.pretrained 
        else:
            for i in self.pretrained.parameters():
                i.requires_grad_(False)
            
            self.pretrained.eval() 
            self.pretrained_tune = None
                

if __name__ == '__main__':     
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    pretrained = AutoModel.from_pretrained("hfl/chinese-bert-wwm-ext")
    path = "D:/03_code/chinese_ner/NER1/data/train_data.csv"
    batch_size = 16
    loader = data_loader(path, batch_size, tokenizer)
    # 获取一个数据样例
    for i, (inputs, labels) in enumerate(loader):
        if i==0:
            example = inputs 
            break 
        
    model = NER(pretrained)
    
    print("模型参数量为",sum(i.numel() for i in pretrained.parameters()) / 10000, "万.")  # 万为单位
    print("模型测算：", pretrained(**inputs).last_hidden_state.shape)  # 最后一维度输出的形状
    print("模型输出的维度：",model(inputs).shape)


