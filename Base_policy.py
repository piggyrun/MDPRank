import numpy as np
import math

class BasePolicy(object):
    def __init__(self, feature_list):
        self.fnum = len(feature_list)
        self.feature_list = feature_list
        self.feature_set = {}
        for f in feature_list:
            self.feature_set.setdefault(f)

    def calcGrad(self, a, doc_list, exp_list, status):
        pass

    def update(self, doc_list, episode, G, gamma, eta):
        pass

    def predict(self, input):
        pass
