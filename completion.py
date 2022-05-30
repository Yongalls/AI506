import csv
import os
import numpy as np
import math
import argparse

from algorithms import logistic_regression, embedding_mlp

from utils import load_data
from embedding import load_embedding, load_embedding_1

# fixed
NUM_ING = 6714

parser = argparse.ArgumentParser()
parser.add_argument("algorithm", help="Name of algorithm")
parser.add_argument("--embedding", default='none', help="Name of embedding")

args = parser.parse_args()

trains, val_cls_q, val_cls_a, val_cpt_q, val_cpt_a, test_cls_q, test_cpt_q = load_data('data/')

if args.embedding == 'onehot':
    embedding = np.identity(NUM_ING)
elif args.embedding != 'none':
    embedding = load_embedding('embeddings/' + args.embedding + '.csv')

# embp1q10 = load_embedding_1('Embedding/Embp1q10.csv', 64)
# embp1q100 = load_embedding_1('Embedding/Embp1q2.csv', 64)


if args.algorithm == 'logistic_regression':
    train_acc, val_acc, val_predict, test_predict = logistic_regression.run(trains, val_cpt_q, val_cpt_a, test_cpt_q, embedding)
elif args.algorithm == 'embedding_mlp':
    train_acc, val_acc, val_predict, test_predict = embedding_mlp.run(trains, val_cpt_q, val_cpt_a, test_cpt_q, embedding, use_neptune=False, hidden_size=512)
    
print("train_acc: %.5f, val_acc: %.5f" % (train_acc, val_acc))
    
savepath = os.path.join('results', args.algorithm)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    
np.savetxt(os.path.join(savepath, 'val_my' + args.embedding + '.csv'), val_predict.astype(int), fmt='%s', delimiter=",")
np.savetxt(os.path.join(savepath, 'test_my' + args.embedding + '.csv'), test_predict.astype(int), fmt='%s', delimiter=",")
    
