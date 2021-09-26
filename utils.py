# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word, iftest=False):
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    if iftest:
        test = load_dataset(config.test_path, config.pad_size)
        return vocab, test
    else:
        train = load_dataset(config.train_path, config.pad_size)
        dev = load_dataset(config.dev_path, config.pad_size)
        return vocab, train, dev


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

# csv数据文件转txt文件，并生成class.txt，部分模型（除LM_cls之外的模型）的输入数据为“文本\tab标签编号”形式
def csv2txt():
    import pandas as pd
    train_data = pd.read_csv("./data/train.csv",header=0).dropna() # 默认with headers，数据格式为text,label
    val_data = pd.read_csv("./data/val.csv",header=0).dropna()
    test_data = pd.read_csv("./data/test.csv",header=0).dropna()
    labels = set(train_data.iloc[:,1])
    assert labels == set(val_data.iloc[:,1])
    with open("./data/class.txt",'w') as f:
        for i in labels:
            f.write("{}\n".format(i))
    labels_dict = {k:v for v,k in enumerate(labels)}
    train_data.replace({list(train_data)[1]:labels_dict}, inplace=True)
    val_data.replace({list(val_data)[1]:labels_dict}, inplace=True)
    test_data.replace({list(test_data)[1]:labels_dict}, inplace=True)
    
    train_data.to_csv('./data/train.txt', index=False, sep='\t', header=False)
    val_data.to_csv('./data/dev.txt', index=False, sep='\t', header=False)
    test_data.to_csv('./data/test.txt', index=False, sep='\t', header=False)

# 将txt数据文件（text \t tag编号）转换成带header的csv文件（text, tag文字）
# 用于数据预处理
def txt2csv():
    import pandas as pd
    data_path = "data/THUCNews/data/"
    train_data = pd.read_csv(data_path + "train.txt",header=None,sep='\t').dropna() # 默认with headers，数据格式为text,label
    val_data = pd.read_csv(data_path + "dev.txt",header=None,sep='\t').dropna()
    test_data = pd.read_csv(data_path + "test.txt",header=None,sep='\t').dropna()
    labels = set(train_data.iloc[:,1])
    assert labels == set(val_data.iloc[:,1])
    labels = list(line.strip() for line in open(data_path + "class.txt",'r'))
    labels_dict = {k:v for k,v in enumerate(labels)}

    train_data.replace({list(train_data)[1]:labels_dict}, inplace=True)
    val_data.replace({list(val_data)[1]:labels_dict}, inplace=True)
    test_data.replace({list(test_data)[1]:labels_dict}, inplace=True)
    
    train_data.columns = ['text','tag']
    val_data.columns = ['text','tag']
    test_data.columns = ['text','tag']
    train_data.to_csv(data_path + "train.csv", index=False)
    val_data.to_csv(data_path + "dev.csv", index=False)
    test_data.to_csv(data_path + "test.csv", index=False)

if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    
    train_dir = "./data/THUCNews/data/train.txt"
    vocab_dir = "./data/THUCNews/data/vocab.pkl"
    pretrain_dir = "./data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    
    # txt2csv()
