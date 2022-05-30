import numpy as np
import os
import networkx as nx
import pandas as pd
import csv
import argparse
from sklearn.decomposition import  TruncatedSVD

from utils import load_data, load_embedding_1


NUM_ING = 6714

def make_embedding(trains, embedding_size):
    G = nx.Graph()
    for i in range(NUM_ING):
        G.add_node(str(i))

    for data in trains:
        for i in range(len(data) - 2):
            for j in range(i+1, len(data) - 1):
                if G.has_edge(data[i], data[j]):
                    G[data[i]][data[j]]['weight'] += 1
                else:
                    G.add_edge(data[i], data[j], weight=1)

    print("Number of nodes: ", G.number_of_nodes(), ", Number of edges: ", G.number_of_edges())
    A = nx.adjacency_matrix(G)
    svd = TruncatedSVD(n_components=embedding_size)
    X = svd.fit_transform(A)
    
    return X
    
    
def load_embedding_1(filename, hiddensize):
    f = open(filename, 'r', encoding='utf-8')
    reader = csv.reader(f)
    f.readline()
    embedding = np.zeros((6714, hiddensize))
    for line in reader:
        i = int(line[1])
        j = 0
        for node in line[2][2:-1].split(' '):
            if node != '':
                embedding[i][j] = float(node.strip())
                j += 1

    f.close()
    return embedding

def load_embedding(filename):
    f = open(filename, 'r', encoding='utf-8')
    reader = csv.reader(f)
    lines = []
    for line in reader:
        lines.append(line)
    
    embedding_size = len(lines[0])
    embedding = np.zeros((NUM_ING, embedding_size))
    for i, line in enumerate(lines):
        for j, value in enumerate(line):
            embedding[i][j] = float(value)
    return embedding

def save_embedding(filename, embedding):
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')
    path = os.path.join('embeddings', filename)
    np.savetxt(path, embedding, delimiter=',') 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, required=True, help="length of embedding")
    args = parser.parse_args()
    
    trains, val_cls_q, val_cls_a, val_cpt_q, val_cpt_a, test_cls_q, test_cpt_q = load_data('data/')
    embedding = make_embedding(trains, args.length)
    print(embedding.shape)
    save_embedding('svd' + str(args.length) + '.csv', embedding)