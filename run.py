# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer, LM')
parser.add_argument('--action_typ', type=str, required=True, help='Provide an action type: train, test')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':

    dataset = 'data/THUCNews'  # 数据集
    
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    action_typ = args.action_typ

    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_name in ['TextCNN','TextRNN','TextRCNN','TextRNN_Att','DPCNN','Transformer']:
        from utils import build_dataset, build_iterator, get_time_dif

    if model_name in ['TextCNN','TextRNN','TextRCNN','TextRNN_Att','DPCNN','Transformer','FastText']:
        from train_eval import train, test, init_network
        x = import_module('models.' + model_name)
        config = x.Config(dataset, embedding)
        if action_typ == 'train':
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True  # 保证每次结果一样
            
            start_time = time.time()
            print("Loading data...")
            vocab, train_data, dev_data = build_dataset(config, args.word, iftest=False)
            train_iter = build_iterator(train_data, config)
            dev_iter = build_iterator(dev_data, config)
            time_dif = get_time_dif(start_time)
            print("Time usage:", time_dif)

            # train
            config.n_vocab = len(vocab)
            model = x.Model(config).to(config.device)
            if model_name != 'Transformer':
                init_network(model)
            print(model.parameters)
            train(config, model, train_iter, dev_iter)
        elif action_typ == 'test':
            vocab, test_data = build_dataset(config, args.word, iftest=True)
            test_iter = build_iterator(test_data, config)
            model = x.Model(config).to(config.device)
            test(config, model, test_iter)
        else:
            raise Exception("Action type not in the list")
    elif model_name == 'LM':
        if action_typ == 'train':
            from models.LM_fc import train
            train()
        elif action_typ == 'test':
            from models.LM_fc import test
            test()
        else:
            raise Exception("Action type not in the list")
    else:
        raise Exception("Model name not in the list")

