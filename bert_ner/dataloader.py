import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from preprocessing import DataPreprocess
from transformers import AutoTokenizer 
from opt import get_opts

class Dataset(Dataset):
    def __init__(self, labels, texts):
        dataset = {
            "tokens": texts,
            "ner_tags": labels
        }
        self.dataset = dataset
            
    def __len__(self):
        return len(self.dataset["tokens"])
    
    def __getitem__(self, i):
        tokens = self.dataset["tokens"][i]
        labels = self.dataset["ner_tags"][i]
        
        return tokens, labels

class collater():
    def __init__(self, tokenizer, hparams):
        self.tokenizer = tokenizer 
        self.hparams = hparams 
    
    def __call__(self, data):
        """
        function: 对tokens进行ids编码，以及序列化，paddings等操作
        outputs: 输出经过padding的tokens,以及tokens的labels
        """
        tokens = [list(i[0]) for i in data]
        labels = [i[1] for i in data]
        inputs = self.tokenizer.batch_encode_plus(tokens,
                                            padding="max_length",  # 补齐tensor长度，以最大长度为准
                                            truncation=True,
                                            max_length=self.hparams.max_length,
                                            return_tensors="pt",
                                            is_split_into_words=True)
        lens = inputs["input_ids"].shape[1]
        # print("----------------", inputs["input_ids"].shape)
        # print("----------------", inputs["attention_mask"].shape)
        # 对label也进行补齐 用0表示padding的位置
        for i in range(len(labels)):  # 对每个label都补齐
            labels[i] = [0] + labels[i]   # 开头补一个cls
            labels[i] += [0] * lens       # 保证长度足够
            labels[i] = labels[i][:lens]  # 只截取最长长度
        
        return inputs, torch.LongTensor(labels) 


def data_loader(hparams, tokenizer):
    dp = DataPreprocess(hparams.root_dir)
    lab_list, text_list = dp.label_text() 
    dataset = Dataset(lab_list, text_list)
    print("The length of dataset is: ", len(dataset), " \n the first element of dataset is:", dataset[0])
    print("The first sequence in dataset is: ", dataset[0][0])
    print("The first label of the first sequence is: ", dataset[0][1])

    # 数据加载
    collate_fn = collater(tokenizer=tokenizer, hparams=hparams)
    loader = DataLoader(dataset,
                        batch_size=hparams.batch_size,
                        collate_fn=collate_fn,  
                        shuffle=True,
                        drop_last=True)

    # 查看数据的样例
    for i, (inputs, labels) in enumerate(loader):
        if i==0:
            print("The number of batchs are ", len(loader)) # 打印batch有多少个batch
            print("The first sequence is ", tokenizer.decode(inputs["input_ids"][0]))
            print("The first label is ", labels[0])

    return loader

if __name__ == "__main__":  
    hparams = get_opts()
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # path = "./data/train_data.json"
    loader = data_loader(hparams, tokenizer)
    