import csv
import numpy as np

def load_data(data_dir):
    print("---------- prepare dataset ----------")
    f = open(data_dir + 'train.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    trains = []
    for line in reader:
        trains.append(line)
    f.close()    
    print("trains", len(trains))

    f = open(data_dir + 'node_ingredient.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    node_ingredients = []
    for line in reader:
        node_ingredients.append(line)
    f.close()    
    print("node_ingredients", len(node_ingredients))
    num_ing = len(node_ingredients)

    f = open(data_dir + 'validation_classification_question.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    val_cls_q = []
    for line in reader:
        val_cls_q.append(line)
    f.close()
    print("val_cls_q", len(val_cls_q))

    f = open(data_dir + 'validation_classification_answer.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    val_cls_a = []
    for line in reader:
        val_cls_a.append(line)
    f.close()
    print("val_cls_a", len(val_cls_a))

    f = open(data_dir + 'validation_completion_question.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    val_cpt_q = []
    for line in reader:
        val_cpt_q.append(line)
    f.close()
    print("val_cpt_q", len(val_cpt_q))

    f = open(data_dir + 'validation_completion_answer.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    val_cpt_a = []
    for line in reader:
        val_cpt_a.append(line)
    f.close()
    print("val_cpt_a", len(val_cpt_a))
    
    f = open(data_dir + 'test_classification_question.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    test_cls_q = []
    for line in reader:
        test_cls_q.append(line)
    f.close()
    print("test_cls_q", len(test_cls_q))
    
    f = open(data_dir + 'test_completion_question.csv', 'r', encoding='utf-8')
    reader = csv.reader(f)
    test_cpt_q = []
    for line in reader:
        test_cpt_q.append(line)
    f.close()
    print("test_cpt_q", len(test_cpt_q))
    print("-------------------------------------\n")
    
    return trains, val_cls_q, val_cls_a, val_cpt_q, val_cpt_a, test_cls_q, test_cpt_q


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count