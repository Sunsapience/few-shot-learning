import matplotlib.pyplot as plt
import numpy as np
import pylab
import random
import pickle
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("D:/tmp/data/", one_hot=True)

# batch_x, batch_y = mnist.train.next_batch(70001)

x_1 = mnist.train.images[:].reshape((-1,28, 28))
x_2 = mnist.test.images[:].reshape((-1,28, 28)) 
y_1 = mnist.train.labels[:]
y_2 = mnist.test.labels[:] 

data = {}
# 建立字典，字典中有10个空列表，存储每个数据
for i in range(10):
    data['c_'+str(i)]=[]

# 把每个类别的数据分别存到上述的10个空列表
for i in range(10):
    for j in range(len(x_1)):
        if np.argmax(y_1[j])==i:
            data['c_'+str(i)].append((x_1[j],i))
    # for j in range(len(x_2)):
    #     if np.argmax(y_2[j])==i:
    #         data['c_'+str(i)].append((x_2[j],i))

d_1 = [] # 训练数据 ,类别 [0,6,8,9,5,4]
d_2 = [] # 测试数据 ,类别 [0,6,8,9,5,4]
d_3 = [] # 测试数据 ,类别 [1,2,3,7]
# l = [len(data['c_'+str(i)]) for i in range(10)]
# print(l)
for i in range(10):
    order = list(range(4500))
    random.shuffle(order)

    # if i<=5:
    #     d_1.extend([data['c_'+str(i)][o] for o in order[:500]])
    #     d_2.extend([data['c_'+str(i)][o] for o in order[3500:]])
    # else:
    #     d_3.extend([data['c_'+str(i)][o] for o in order[3500:]])
    if i in [0,6,8,9,5,4]:
        d_1.extend([data['c_'+str(i)][o] for o in order[:500]])
        d_2.extend([data['c_'+str(i)][o] for o in order[3500:]])
    else:
        d_3.extend([data['c_'+str(i)][o] for o in order[3500:]])

# print(d_1[0])
with open(r'D:\wlgzg\Documents\workshop_\few shot learning\data\Data.pkl','wb') as f:
    pickle.dump((d_1,d_2,d_3),f)
