import sys
import random
import numpy as np

class DataManger:
    def __init__(self, train_rate = 0.8):
        self.train_rate = train_rate
        self.test_rate = (1 - self.train_rate) / 2 + self.train_rate

    def getDataRange(self, datafile, range_file):
        frange_map = {}
        fin = open(datafile, "r")
        for line in fin:
            endi = line.find("#")
            if endi < 0:
                continue;
            tokens = line[:endi].split(" ")
            query = tokens[1]
            for i in xrange(2, len(tokens)-1):
                fid_v = tokens[i].split(":")
                if len(fid_v) == 2:
                    fid = fid_v[0]
                    val = float(fid_v[1])
                    if frange_map.has_key(fid):
                        (fmin, fmax) = frange_map[fid]
                        if val > fmax:
                            fmax = val
                        if val < fmin:
                            fmin = val
                        frange_map[fid] = (fmin, fmax)
                    else:
                        frange_map[fid] = (val, val)
        fin.close()

        fout = open(range_file, "w")
        for (fid, vals) in frange_map.items():
            fout.write("%s\t%.2f\t%.2f\n"%(fid, vals[0], vals[1]))
        fout.close()

    def genData(self, datafile, range_file, train_file, valid_file, test_file):
        frange_map = {}
        frg = open(range_file, "r")
        for line in frg:
            tokens = line.split("\t")
            if len(tokens) == 3:
                fid = tokens[0]
                fmin = float(tokens[1])
                fmax = float(tokens[2])
                frange_map.setdefault(fid, (fmin, fmax))
        frg.close()

        data_map = {}
        fin = open(datafile, "r")
        for line in fin:
            endi = line.find("#")
            if endi < 0:
                continue;
            tokens = line[:endi].split(" ")
            query = tokens[1]
            norm_line = "%s %s "%(tokens[0], query)
            is_bad = False
            for i in xrange(2, len(tokens)-1):
                feature = tokens[i]
                fid_v = feature.split(":")
                if len(fid_v) == 2:
                    fid = fid_v[0]
                    val = float(fid_v[1])
                    if frange_map.has_key(fid):
                        fvals = frange_map[fid]
                        val = self.normalize(val, fvals[0], fvals[1])
                    else:
                        print >>sys.stderr, "feature fid (%s) missed!"%(fid)
                    if val < 0: # abnormal data
                        is_bad = True
                        print >>sys.stderr, "bad feature fid(%s): %s"%(fid, line.rstrip())
                        break

                    feature = "%s:%.2f"%(fid, val)
                else:
                    print >>sys.stderr, "feature (%s) format error!"%(feature)
                norm_line += feature + " "

            if not is_bad:
                norm_line += line[endi:]
                if data_map.has_key(query):
                    data_map[query].append(norm_line)
                else:
                    data_map.setdefault(query, [norm_line])
        fin.close()

        ftrain = open(train_file, "w")
        fvalid = open(valid_file, "w")
        ftest = open(test_file, "w")
        fout = ftrain
        for q,v in data_map.items():
            r = random.random()
            if r < self.train_rate:
                fout = ftrain
            elif r < self.test_rate:
                fout = ftest
            else:
                fout = fvalid
            for line in v:
                fout.write(line)
        ftrain.close()
        fvalid.close()
        ftest.close()

    def normalize(self, v, fmin, fmax):
        if v > fmax:
            return -1
        elif v < fmin:
            return -1
        elif fmax - fmin < 1e-3:
            return 0.0
        else:
            return (v - fmin) / (fmax - fmin)


if __name__ == "__main__":
    dm = DataManger()
    #dm.getDataRange("training_samples", "feature_range.raw")
    dm.genData("training_samples", "feature_range", "train_data", "valid_data", "test_data")
                

