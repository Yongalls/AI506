import numpy as np
from sklearn.linear_model import LogisticRegression

NUM_ING = 6714

def make_trainset(trainset, embedding):
    xs = []
    ys = []
    for data in trainset:
        nodes = [int(node) for node in data[:-1]]
        for i, node in enumerate(nodes):
            other_nodes = [nd for nd in nodes if nd != node]
            if len(other_nodes) == 0:
                continue
            x = np.zeros(embedding.shape[1])
            y = node
            for other in other_nodes:
                x += embedding[other]
            if embedding.shape[1] != NUM_ING:
                x = x / len(other_nodes)
            xs.append(x)
            ys.append(y)
    train_x = np.stack(xs, axis=0)
    train_y = np.stack(ys, axis=0)
    return train_x, train_y


def make_valset(question, answer, embedding):
    xs = []
    ys = []
    for i in range(len(question)):
        nodes = [int(node) for node in question[i]]
        y = int(answer[i][0])
        x = np.zeros(embedding.shape[1])
        for node in nodes:
            x += embedding[node]
        if embedding.shape[1] != NUM_ING:
            x = x / len(nodes)
        xs.append(x)
        ys.append(y)
    val_x = np.stack(xs, axis=0)
    val_y = np.stack(ys, axis=0)
    return val_x, val_y

def make_testset(testset, embedding):
    xs = []
    for data in testset:
        nodes = [int(node) for node in data]
        x = np.zeros(embedding.shape[1])
        for node in nodes:
            x += embedding[node]
        if embedding.shape[1] != NUM_ING:
            x = x / len(nodes)
        xs.append(x)
    test_x = np.stack(xs, axis=0)
    return test_x

def run(trainset, valset_q, valset_a, testset, embedding):
    print("run logistic")
    train_x, train_y = make_trainset(trainset, embedding)
    print("train: ", train_x.shape, train_y.shape)
    val_x, val_y = make_valset(valset_q, valset_a, embedding)
    print("val: ", val_x.shape, val_y.shape)
    test_x = make_testset(testset, embedding)
    print("test: ", test_x.shape)
    
    clf = LogisticRegression(penalty='l2', max_iter=20, verbose=True).fit(train_x, train_y)
    train_acc = clf.score(train_x, train_y)
    val_acc = clf.score(val_x, val_y)
        
    val_predict = clf.predict(val_x)
    test_predict = clf.predict(test_x)
    return train_acc, val_acc, val_predict, test_predict
    