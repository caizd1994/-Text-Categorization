import tensorflow as tf
import numpy as np
import pandas as pd


data = pd.read_csv("data.csv")



writer = tf.python_io.TFRecordWriter("train.tfrecords")
train = data[data['target'] != -1]

for index, row in train.iterrows():
    # print(row['target'])
    temp = [0 for i in range(0,208)]
    for index,elem in enumerate(row['question_text'].strip("[]").split(",")):
        temp[index] = int(elem)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row['target'])])),
                'word': tf.train.Feature(int64_list=tf.train.Int64List(value=temp))
            }
        )
    )
    writer.write(example.SerializeToString())
writer.close()


writer = tf.python_io.TFRecordWriter("test.tfrecords")
test = data[data['target'] == -1]
for index, row in train.iterrows():
    # print(row['target'])
    temp = [0 for i in range(0,208)]
    for index,elem in enumerate(row['question_text'].strip("[]").split(",")):
        temp[index] = int(elem)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'word': tf.train.Feature(int64_list=tf.train.Int64List(value=temp))
            }
        )
    )
    writer.write(example.SerializeToString())
writer.close()