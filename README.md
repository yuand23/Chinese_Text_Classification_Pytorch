## 使用说明
```
# 训练：
# TextCNN
python run.py --model TextCNN --action_typ train

# TextRNN
python run.py --model TextRNN --action_typ train

# TextRNN_Att
python run.py --model TextRNN_Att --action_typ train

# TextRCNN
python run.py --model TextRCNN --action_typ train

# FastText, embedding层是随机初始化的
python run.py --model FastText --embedding random --action_typ train

# DPCNN
python run.py --model DPCNN --action_typ train

# Transformer
python run.py --model Transformer --action_typ train

# Pretrained LM
python run.py --model LM --action_typ train
```

```
# 测试：
# TextCNN
python run.py --model TextCNN --action_typ test

# TextRNN
python run.py --model TextRNN --action_typ test

# TextRNN_Att
python run.py --model TextRNN_Att --action_typ test

# TextRCNN
python run.py --model TextRCNN --action_typ test

# FastText, embedding层是随机初始化的
python run.py --model FastText --embedding random --action_typ test

# DPCNN
python run.py --model DPCNN --action_typ test

# Transformer
python run.py --model Transformer --action_typ test

# Pretrained LM
python run.py --model LM --action_typ test
```

```
# 交互式测试：
# Pretrained LM
python run.py --model LM --action_typ test_single
```

生成的测试文件保存于test_result.txt  

数据格式1：train.txt, dev.txt, test.txt without headers    
text \t tag编号  
以及class.txt，为类别名称  

数据格式2：train.csv, dev.csv, test.csv with headers  
text,tag名称  
使用python run.py --model preprosses --action_typ preprosses转换成格式1  

## referece
https://github.com/649453932/Chinese-Text-Classification-Pytorch  
增加了预训练语言模型+fc分类模型；将测试和训练分开，并输出测试结果。  



