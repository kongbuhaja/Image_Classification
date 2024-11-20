import numpy as np

class Metric():
    def __init__(self, nc, macro=True, eps=1e-6):
        self.tp = np.zeros(nc) + eps
        self.fp = np.zeros(nc) + eps
        self.fn = np.zeros(nc) + eps
        self.macro = macro

    def update(self, pred, label):
        for i in range(len(label)):
            if label[i] == pred[i]:
                self.tp[label[i]] += 1 
            else:
                self.fp[pred[i]] += 1 
                self.fn[label[i]] += 1 

    def recall(self):
        if self.macro:
            return np.mean(self.tp / (self.tp + self.fn))
        else:
            return np.sum(self.tp) / (np.sum(self.tp) + np.sum(self.fn))

    def precision(self):
        if self.macro:
            return np.mean(self.tp / (self.tp + self.fp))
        else:
            return np.sum(self.tp) / (np.sum(self.tp) + np.sum(self.fp))

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall)
    
    def stat(self):
        precision = self.precision()
        recall = self.recall()
        return recall, precision, 2 * (precision * recall) / (precision + recall)