import numpy as np
import math
from Base_policy import *

class SoftmaxPolicy(BasePolicy):
    def __init__(self, feature_list):
        super(SoftmaxPolicy, self).__init__(feature_list)
        self.w = np.zeros(self.fnum)

    def load_model(self, model_file):
        fin = open(model_file, "r")
        idx = 0
        for line in fin:
            if idx >= self.fnum:
                print >>sys.stderr, "model dimension mismatch!"
                break

            tokens = line.rstrip().split("\t")
            fid = tokens[0]
            wt = float(tokens[1])
            if self.feature_set.has_key(fid):
                self.w[idx] = wt
                idx += 1
        fin.close()

    def save_model(self, model_file):
        fout = open(model_file, "w")
        for i in xrange(self.fnum):
            fout.write("%s\t%.4f\n"%(self.feature_list[i], self.w[i]))
        fout.close()

    def calcGrad(self, a, doc_list, exp_list, status):
        size = len(doc_list)
        Vmax = float("-inf")
        exp_v = np.zeros(size)
        for i in xrange(size):
            if status.has_key(i):
                exp_v[i] = exp_list[i]
                if exp_v[i] > Vmax:
                    Vmax = exp_v[i]

        fa = np.array(doc_list[a][0])
        ftmp = np.zeros(fa.shape)
        pro_sum = 0.0
        for i in status.keys():
            fi = np.array(doc_list[i][0])
            pro = np.exp(exp_v[i] - Vmax)
            ftmp += fi * pro
            pro_sum += pro
        grd = fa - ftmp / pro_sum
        return grd

    def update(self, doc_list, exp_list, episode, G, gamma, eta):
        M = len(doc_list)
        status = {}
        for i in xrange(M):
            status.setdefault(i)

        deltaW = np.zeros(self.w.shape)
        for i in xrange(M-1):
            st = episode[i]
            a = st[0]
            grd = self.calcGrad(a, doc_list, exp_list, status)
            deltaW += math.pow(gamma, i) * G[i] * grd
            del status[a]

        self.w += eta * deltaW

    def predict(self, doc_list):
        size = len(doc_list)
        ys = np.zeros(size)
        for i in xrange(size):
            ys[i] = np.dot(doc_list[i][0], self.w)
        return ys

    def Pi(self, exp_list, status):
        size = len(exp_list)
        Vmax = float("-inf")
        exp_v = np.zeros(size)
        for i in xrange(size):
            if status.has_key(i):
                exp_v[i] = exp_list[i]
                if exp_v[i] > Vmax:
                    Vmax = exp_v[i]

        pi = np.zeros(size)
        curr_sum = 0
        for i in xrange(size):
            if status.has_key(i):
                pro = np.exp(exp_v[i] - Vmax)
            else:
                pro = 0
            pi[i] = pro
            curr_sum += pro
        pi /= curr_sum
        return pi
