

#对计算结果和label变形,并且移除pad
def reshape_and_remove_pad(outs, labels, attention_mask):
    #变形,便于计算loss
    #[b, lens, 7] -> [b*lens, 7]
    outs = outs.reshape(-1, 7)
    #[b, lens] -> [b*lens]
    labels = labels.reshape(-1)

    #忽略对pad的计算结果
    #[b, lens] -> [b*lens - pad]
    select = attention_mask.reshape(-1) == 1
    outs = outs[select]
    labels = labels[select]

    return outs, labels


#获取正确数量和总数
def get_correct_and_total_count(labels, outs):
    #[b*lens, 7] -> [b*lens]
    outs = outs.argmax(dim=1)
    print("outs shape = ", outs.shape)
    print("outs = ", outs[:200])
    correct = (outs == labels).sum().item()
    total = len(labels)

    #计算除了0以外元素的正确率,因为0太多了,包括的话,正确率很容易虚高
    select = labels != 0
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs == labels).sum().item()
    total_content = len(labels)

    return correct, total, correct_content, total_content




if __name__ == "__main__":
    pass