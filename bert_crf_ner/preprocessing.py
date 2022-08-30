import pandas as pd
import json
import re
import random
from opt import get_opts

class DataPreprocess():
    def __init__(self, path):
        self.path = path 
        # self.data = None 
        self.data = None
    
    def read_data(self):
        # 读取数据
        # with open(self.path, 'r', encoding="utf-8") as f:
        #     self.data = json.load(f)
        json_data = []
        for line in open(self.path, 'r', encoding='utf-8'):
            json_data.append(json.loads(line))

        self.data = json_data

        return json_data


    def label_text(self):
        data = self.read_data()
        text_l = []
        target_l = []  
        idx_l = []
        att_l = []  # attribute 
        label_l = []   # {'definition', 'experimental_result', 'None'}
        for i, instance in enumerate(data):
            text = instance["src"][27:]
            # 句子太长的处理： 512丢弃
            if len(list(text)) > 512:
                continue 
            text = text.replace("(", "（").replace(")", "）")
            text_l.append(text) 
            tgt = instance["tgt"]
            # tgt = tgt.replace("(", "（").replace(")", "）").replace("+", "\\+")
            tgt = tgt.translate(str.maketrans({
                                            "(": "（",
                                            ")": "）",
                                            "+": "\\+",
                                            "{": "\\{",
                                            "}": "\\}",
                                            "^": "\\^",
                                            "[": "\\[",
                                            "]": "\\]", 
                                            "?": "\\?",
                                            "*": "\\*",
                                            ".": "\\.",
                                            "\\": "/",
                                            }))
            # ( ,), +识别不了
            
            att = re.findall(r"[a-zA-Z_]+", tgt)[0]
            att_l.append(att)
            target = re.findall(r"^.+: (.+)$", tgt)[0]
            target_l.append(target)   
            if att == "None": 
                idx = 0
                idx_l.append(idx)
                label = [0] * len(text)
                label_l.append(label)
            else:
                match_idx = re.search(target, text).span()
                start = match_idx[0]
                end = match_idx[1]
                idx = (start, end)
                idx_l.append(idx)
                label = [0] * len(text)
                if att == "definition": 
                    label[start:end] = [1] * (end-start)
                    label_l.append(label)
                else:   
                    label[start:end] = [2] * (end-start)
                    label_l.append(label)
                
                # print(text[start:end])
        print("="*50)
        print("length of attributes", len(att_l))
        print("length of indexs", len(idx_l))
        print("length of labels", len(label_l))
        print("="*50)
            
        self.lab_list = label_l
        self.text_list = text_l
        return self.lab_list, self.text_list
    
    def train_test_split(self):
        data = self.read_data()
        lens = len(data)
        data = random.sample(data, lens) 
        train_lens = int(lens * 0.8)
        train_data = data[:train_lens]
        test_data = data[train_lens:]

        with open("./data/train_data.json", "w", encoding="utf-8") as f:
            f.write(
                '\n'.join(json.dumps(i, ensure_ascii=False) for i in train_data) 
            )
            
        with open("./data/test_data.json", "w", encoding="utf-8") as f:
            f.write(
                '\n'.join(json.dumps(i, ensure_ascii=False) for i in test_data) 
            )
            

if __name__ == '__main__':   
    hparams = get_opts()
    #path = "D:/03_code/chinese_ner/NER2/data/train_multi.json"
    # path = "D:/03_code/chinese_ner/NER2/data/train_data.json"
    # path = "D:/03_code/chinese_ner/NER2/data/test_data.json"
    dp = DataPreprocess(hparams.root_dir)
    data = dp.read_data()
    dp.train_test_split()
    lab_list, text_list = dp.label_text()
    print("lab example：", lab_list[0])
    print("text example：", text_list[0])
    
    lens = []
    for i in text_list:
        l = len(i)
        lens.append(l)
        
    print("序列的最大长度", max(lens))
