from collections import defaultdict
import numpy as np
from tqdm import tqdm
import time

NUM_ING = 6714

def run(trains, val_cpt_q, val_cpt_a, lamda=0.25, threshold = 10):

    singles = defaultdict(int)
    # pairs = defaultdict(int)
    triples = defaultdict(int)
    quadraples = defaultdict(int)

    for data in tqdm(trains):
        nodes = [int(node) for node in data[0:-1]]
        nodes = sorted(nodes)
        for i in range(len(nodes)):
            singles[nodes[i]] += 1

        # for i in range(len(nodes) - 1):
        #     for j in range(i+1, len(nodes)):
        #         pairs[(nodes[i], nodes[j])] += 1

        for i in range(len(nodes) - 2):
            for j in range(i+1, len(nodes) - 1):
                for k in range(j+1, len(nodes)):
                    triples[(nodes[i], nodes[j], nodes[k])] += 1

        for i in range(len(nodes) - 3):
            for j in range(i+1, len(nodes) - 2):
                for k in range(j+1, len(nodes) - 1):
                    for l in range(k+1, len(nodes)):
                        quadraples[(nodes[i], nodes[j], nodes[k], nodes[l])] += 1


    def interest(I, j, lamda):
        confidence = quadraples[tuple(sorted(I + [j]))] / (triples[tuple(sorted(I))] + 0.01)
        return abs(confidence - lamda * singles[j] / len(trains))

    acc = 0
    estimations = []
    start = time.time()
    for id, data in tqdm(enumerate(val_cpt_q)):
        nodes_temp = [int(node) for node in data]
        nodes = [node for node in nodes_temp if singles[node] > threshold]
    #     print("nodes: ", len(nodes_temp), len(nodes))
        target = int(val_cpt_a[id][0])
        interest_list = [0] * NUM_ING
        len_node = len(nodes)
        for ing in range(NUM_ING):
            if ing in nodes_temp:
                interest_list[ing] = 0
            else:
                for i in range(len_node - 2):
                    for j in range(i+1, len_node - 1):
                        for k in range(j+1, len_node):
                            interest_list[ing] += interest([nodes[i], nodes[j], nodes[k]], ing, lamda)

        estimation = max(range(len(interest_list)), key=lambda i: interest_list[i])
    #     print(estimation==target, max(interest_list), interest_list[target], nodes_temp, estimation, target)
        estimations.append(estimation)

        if estimation == target:
            acc += 1

        if id % 100 == 0:
            print(acc / (id+1))

    print(time.time() - start)
    estimations = np.array(estimations)
    return acc / len(val_cpt_a), estimations
