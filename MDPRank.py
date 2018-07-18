import sys
import random
import math
import re
import numpy as np
from Softmax_policy import *
from DNN_policy import *
import time

class MDPRank:
    def __init__(self, feature_file, gamma = 1.0, eta = 0.05, est_type = "mc"):
        ffea = open(feature_file, "r")
        self.feature_list = []
        self.feature_set = {}
        for line in ffea:
            line = line.rstrip()
            if not re.match("^\s*#", line):
                fi = line.split("\t")[0]
                self.feature_list.append(fi)
                self.feature_set.setdefault(fi)

        self.fnum = len(self.feature_list)
        self.gamma = gamma
        self.eta = eta
        self.est_type = est_type

        #self.policy = SoftmaxPolicy(self.feature_list)
        self.policy = DnnPolicy(self.feature_list)

        self.train_data = {}
        self.valid_data = {}
        self.test_data = {}

        print "feature num: %d, gamma: %.2f, eta: %.2f"%(self.fnum, self.gamma, self.eta)

    def load_sample(self, datafile, data_map):
        data_map.clear()
        fin = open(datafile, "r")
        for line in fin:
            endi = line.find("#")
            if endi < 0:
                continue;
            tokens = line[:endi].split(" ")
            label = tokens[0]
            query = tokens[1]
            fvec = np.zeros(self.fnum)
            idx = 0
            for i in xrange(2, len(tokens)-1):
                fid_v = tokens[i].split(":")
                if self.feature_set.has_key(fid_v[0]):
                    if len(fid_v) == 2:
                        fvec[idx] = float(fid_v[1])
                    else:
                        print "feature format error: ", tokens[i]
                    idx += 1

            if data_map.has_key(query):
                data_map[query].append((fvec, label))
            else:
                data_map.setdefault(query, [(fvec, label)])
        
        print "load data complete! size:", len(data_map)
        fin.close()

    def load_data(self, train_file, valid_file, test_file):
        self.load_sample(train_file, self.train_data)
        self.load_sample(valid_file, self.valid_data)
        self.load_sample(test_file, self.test_data)

    def load_training(self, train_file, valid_file):
        self.load_sample(train_file, self.train_data)
        self.load_sample(valid_file, self.valid_data)

    def load_testing(self, test_file):
        self.load_sample(test_file, self.test_data)

    def load_model(self, model):
        self.policy.load_model(model)

    def save_model(self, model_file):
        self.policy.save_model(model_file)

    def sampleAction(self, doc_list, exp_list, status):
        size = len(doc_list)
        pi = self.policy.Pi(exp_list, status)

        hist = []
        curr_sum = 0
        for i in xrange(size):
            curr_sum += pi[i]
            hist.append(curr_sum)

        r = random.random() * curr_sum
        a = 0
        for i in xrange(size):
            if hist[i] > r:
                a = i
                break

        if not status.has_key(a):
            print >>sys.stderr, "sample action error"
            print >>sys.stderr, "a: ",a," sum: ",curr_sum, " r: ",r
            print >>sys.stderr, "status: ", status
            print >>sys.stderr, "hist: ", hist

        return a

    def sampleAnEpisode(self, doc_list, exp_list):
        size = len(doc_list)
        status = {}
        for i in xrange(size):
            status.setdefault(i)

        episode = []
        for i in xrange(size-1):
            a = self.sampleAction(doc_list, exp_list, status)
            del status[a]
            label = float(doc_list[a][1])
            r = label #self.getReward2(i, label)
            st = (a, r)
            episode.append(st)

        return episode

    def getReward(self, t, label):
        if t == 0:
            r = math.pow(2, label) - 1
        else:
            r = (math.pow(2, label) - 1) * math.log10(2) / math.log10(t+2)
        return r

    def getReward2(self, t, label):
        r = (math.pow(2, label) - 1) / (t + 1)
        return r

    def calcQlearingGt(self, episode, k):
        M = len(episode)
        all_list = [episode[k][1]]
        label_list = []
        if k + 1 < M:
            for i in xrange(k+1, M):
                label_list.append(episode[i][1])
            label_list.sort(reverse=True)
        all_list.extend(label_list)
        
        Gt = 0
        for i in xrange(len(all_list)-1, -1, -1):
            Gt = Gt * self.gamma + self.getReward2(i, all_list[i])

        return Gt

    def calcQlearingG(self, episode):
        M = len(episode)
        G = [0 for i in xrange(M)]
        for i in xrange(M):
            G[i] = self.calcQlearingGt(episode, i)
        return G

    def calcG(self, episode):
        M = len(episode)
        G = [0 for i in xrange(M)]
        Gt = 0
        for i in xrange(M-1, -1, -1):
            Gt = Gt * self.gamma + self.getReward2(i, episode[i][1])
            G[i] = Gt
        return G

    def train(self):
        tr_data = self.train_data.values()
        tr_size = len(tr_data)
        is_finish = False
        idx = 0
        shuf_idx = range(tr_size)
        random.shuffle(shuf_idx)
        batch = 0
        t1 = time.time()
        while not is_finish:
            doc_list = tr_data[shuf_idx[idx]]
            exp_list = self.policy.predict(doc_list)
            M = len(doc_list)
            #print "sample %d(%d) processing ..."%(idx, M)
            epd = self.sampleAnEpisode(doc_list, exp_list)
            #print "episode: ", epd
            if self.est_type == "q":
                G = self.calcQlearingG(epd)
            elif self.est_type == "mc":
                G = self.calcG(epd)
            else:
                G = self.calcG(epd)
            #print "G: ", G
            self.policy.update(doc_list, exp_list, epd, G, self.gamma, self.eta)
            #print "sample %d(%d) done"%(idx, M)

            if idx % 100 == 0:
                t2 = time.time()
                diff_t = t2 - t1
                t1 = t2
                print "training progress (%d/%d) ... cost: %.3fs"%(idx, batch, diff_t) 
            if idx % 1000 == 0:
                self.validate()

            idx += 1
            if (idx >= tr_size):
                idx = 0
                batch += 1
                print "batch: ", batch

            if batch > 3:
                is_finish = True

    def validate(self):
        v_data = self.valid_data.values()
        cnt = 0
        G0 = 0.0
        ndcg = 0.0
        err = 0.0
        for doc_list in v_data:
            exp_list = self.policy.predict(doc_list)
            epd = self.sampleAnEpisode(doc_list, exp_list)
            if self.est_type == "q":
                G = self.calcQlearingG(epd)
            elif self.est_type == "mc":
                G = self.calcG(epd)
            else:
                G = self.calcG(epd)
            G0 += G[0]

            ret = []
            idx = 0
            for doc in doc_list:
                fvec = [doc[0]]
                label = int(doc[1])
                score = exp_list[idx]
                idx += 1
                #ret.append((score, label, fvec))
                ret.append((score, label))
            ret.sort(self.rank_cmp)
            #print "ret: ", ret

            label_list = []
            for i in xrange(len(ret)):
                label_list.append(ret[i][1])
            ndcg += self.NDCG(label_list)
            err += self.ERR(label_list)
            cnt += 1

        G0 /= cnt
        ndcg /= cnt
        err /= cnt
        print "Validation - cnt: %d, G: %.4f, NDCG: %.4f, ERR: %.4f"%(cnt, G0, ndcg, err)

    def rank_cmp(self, x, y):
        if x[0] < y[0]:
            return 1
        elif x[0] > y[0]:
            return -1
        else:
            return 0

    def test(self):
        t_data = self.test_data.values()
        cnt = 0
        ndcg = 0
        err = 0
        for doc_list in t_data:
            ret = []
            ys = self.policy.predict(doc_list)
            idx = 0
            for doc in doc_list:
                fvec = [doc[0]]
                label = int(doc[1])
                score = ys[idx]
                idx += 1
                ret.append((score, label, fvec))
            ret.sort(self.rank_cmp)

            label_list = []
            for i in xrange(len(ret)):
                label_list.append(ret[i][1])
                #print "%d\t%.2f\t%d\t%s"%(i, ret[i][0], ret[i][1], ret[i][2])
            ndcg += self.NDCG(label_list)
            err += self.ERR(label_list)
            cnt += 1

        ndcg /= cnt
        err /= cnt
        print "Result - cnt: %d, NDCG: %.4f, ERR: %.4f"%(cnt, ndcg, err)

    def DCG(self, r, k = 10):
        ranks = np.asfarray(r)[:k]
        dcg = 0
        if ranks.size > 0:
            dcg = np.sum(ranks / np.arange(2, ranks.size + 2))
        return dcg

    def NDCG(self, r, k = 10):
        dcg_max = self.DCG(sorted(r, reverse=True), k)
        ndcg = 0
        if dcg_max > 0:
            dcg = self.DCG(r, k)
            ndcg = dcg / dcg_max
        return ndcg

    def ERR(self, r, k = 10):
        ranks = np.asfarray(r)[:k]
        err = 0
        if ranks.size > 0:
            ranks = (np.power(2, ranks) - 1) / np.power(2, 5)
            for i in xrange(ranks.size):
                v = ranks[i]
                for j in xrange(i):
                    v *= (1 - ranks[j])
                err += v / (i+1)
        return err

