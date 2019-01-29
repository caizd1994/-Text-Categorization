import pandas as pd
import tensorflow as tf
import numpy as np
import re
from tensorflow.contrib import learn
import gensim
import pickle

_PAD="_PAD"
_UNK="UNK"

train = pd.read_csv('../all/train.csv')
test = pd.read_csv('../all/test.csv')
test["target"]=-1
data = pd.concat([train,test],axis=0)
print(data.shape)
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
data['question_text'] = data['question_text'].map(lambda x:clean_str(x))
word_list = {}
cnt = 0
word = []
max_len = 0
list_len = []
for elem in data['question_text']:
    field = elem.split(" ")
    if max_len< len(field):
        max_len = len(field)
        list_len.append(len(field))
    for i in field:
        word.append(i)
word = set(word)
print("max_len",max_len)
max_len = 0
for elem in list_len:
    max_len+=elem
print("mean_len",max_len/len(list_len))
print(len(word))
model = gensim.models.KeyedVectors.load_word2vec_format('../all/embeddings/1/1.bin', binary=True)
embedding = {}
cnt = 1
vocab = model.vocab
print(len(vocab)) #谷歌词向量词数
for elem in word:
    if elem in vocab:
        embedding[elem] = model.get_vector(elem)
        word_list[elem] = cnt
        cnt += 1
    else:
        embedding[elem] = [0 for i in range(300)]
        word_list[elem] = 0
data['question_text'] = data['question_text'].map(lambda x:[word_list[elem] for elem in x.split(" ")])
# print(data['question_text'])
print("embedding_len",len(embedding))
embedding_list = []
for key,value in embedding.items():
    embedding_list.append((key,value))
final = []
cnt = 0
for key,value in embedding.items():
    if (word_list[key] == 0 and cnt ==0):
        final.append((word_list[key],value))
        cnt += 1
    elif (word_list[key] != 0):
        final.append((word_list[key], value))
    else:
        continue
print(final)
print(len(final))

with open('../data/remap.pkl', 'wb') as f:
  pickle.dump(final, f, pickle.HIGHEST_PROTOCOL) # uid, iid
data.to_csv("../data/data.csv",index=False)

