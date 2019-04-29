import pickle,random
import itertools

with open(r'D:\wlgzg\Documents\workshop_\few shot learning\data\Data.pkl','rb') as f:
    data = pickle.load(f)

train_data_,test_1,test_2 = data[0],data[1],data[2]
# train_data  只包含0-5训练数据
# test_1 0-5 测试数据
# test_2 6-9 测试数据
# print(len(train_data))
# print(len(test_1),len(test_2))
def batch_train_flow(batch_size):
    o_ = list(range(len(train_data_)))
    random.shuffle(o_)
    train_data = [train_data_[oo] for oo in o_]

    oo = itertools.combinations(range(len(train_data)),2)
    order = [o for o in oo]
    n_batches = len(order[:int(5e5)]) // batch_size

    train_x_1 = [train_data[i[0]][0] for i in order[:int(5e5)]]
    train_x_2 = [train_data[i[1]][0] for i in order]
    train_y_ = [(train_data[i[0]][1],train_data[i[1]][1]) for i in order[:int(5e5)]]
    train_y = [1 if i[0]==i[1] else 0 for i in train_y_]
    for j in range(n_batches):
        yield train_x_1[j*batch_size:(j+1)*batch_size],train_x_2[j*batch_size:(j+1)*batch_size],train_y[j*batch_size:(j+1)*batch_size]

def batch_test_train():
    return [i[0] for i in train_data_]

def batch_test_same():
    o_ = list(range(len(test_1)))
    random.shuffle(o_)
    # return [i[0] for i in test_1]
    return [test_1[o][0] for o in o_[:3000]]

def batch_test_different():
    return [i[0] for i in test_2]
    # o_ = list(range(len(test_2)))
    # random.shuffle(o_)
    # return [test_2[o][0] for o in o_[:2000]]

# if __name__ == "__main__":
#     oo = itertools.combinations(range(len(train_data)),2)
#     order = [o for o in oo]
#     n_batches = len(order) // 150
#     print(order[3])
#     print(order[3][1])                                          
#     print(len(train_data[4]))
#     train_x_1 = [train_data[i[0]][0] for i in order]
