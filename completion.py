import csv
import os
import numpy as np
import math
import argparse

from utils import load_data
from embedding import load_embedding, load_embedding_1

# fixed
NUM_ING = 6714

parser = argparse.ArgumentParser()
parser.add_argument("algorithm", help="Name of algorithm")
parser.add_argument("--embedding", default='none', help="Name of embedding")
parser.add_argument("--filename", default='result', help="filename")
parser.add_argument("--batchsize", default=256, type=int, help="batchsize")
parser.add_argument("--hiddensize", default=512, type=int, help="hidden size")
parser.add_argument("--epochs", default=150, type=int, help="epochs")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--gpu", default='0', type=str, help="gpu")
parser.add_argument("--neptune", action='store_true')
parser.add_argument("--start", default=0, type=int, help="iteration start")
parser.add_argument("--end", default=7848, type=int, help="iteration end")


args = parser.parse_args()

trains, val_cls_q, val_cls_a, val_cpt_q, val_cpt_a, test_cls_q, test_cpt_q = load_data('data/')

if args.embedding == 'onehot':
    embedding = np.identity(NUM_ING)
elif args.embedding != 'none':
    embedding = load_embedding('embeddings/' + args.embedding + '.csv')

# embp1q10 = load_embedding_1('Embedding/Embp1q10.csv', 64)
# embp1q100 = load_embedding_1('Embedding/Embp1q2.csv', 64)


if args.algorithm == 'logistic_regression':
    from algorithms import logistic_regression
    train_acc, val_acc, val_predict, test_predict = logistic_regression.run(trains, val_cpt_q, val_cpt_a, test_cpt_q, embedding)
    print("train_acc: %.5f, val_acc: %.5f" % (train_acc, val_acc))
elif args.algorithm == 'embedding_mlp':
    from algorithms import embedding_mlp
    train_acc, val_acc, val_predict, test_predict = embedding_mlp.run(trains, val_cpt_q, val_cpt_a, test_cpt_q, embedding, use_neptune=False, hidden_size=512)
    print("train_acc: %.5f, val_acc: %.5f" % (train_acc, val_acc))
elif args.algorithm == 'frequent_items':
    from algorithms import frequent_item
    val_acc, val_predict = frequent_item.run(trains, val_cpt_q[args.start:args.end], val_cpt_a[args.start:args.end])
    print("val_acc: ", val_acc)
else:
    raise Exception("Invalid algorithm name")


savepath = os.path.join('results', args.algorithm)
if not os.path.exists(savepath):
    os.makedirs(savepath)

np.savetxt(os.path.join(savepath, 'val_' + args.filename + '.csv'), val_predict.astype(int), fmt='%s', delimiter=",")
# np.savetxt(os.path.join(savepath, 'test_' + args.filename + '.csv'), test_predict.astype(int), fmt='%s', delimiter=",")
