#coding=utf-8
import tensorflow as tf
import numpy as np
import math
from Base_policy import *

class DnnPolicy(BasePolicy):
    """
    网络结构：输入为doc的特征向量，输出为一个数
    一个session中，单个doc作为输入求导数，得到一个输出，共有n个导数与输出，对输出求softmax得到概率分布PI
    某个actions的梯度为，该doc的导数减去候选doc导数的softmax概率和
    """
    def __init__(self, feature_list):
        super(DnnPolicy, self).__init__(feature_list)
        
        x = tf.placeholder(tf.float32, [None, self.fnum]) # features of an action
        self.input_x = x

        # Set model weights
        hidden_num = self.fnum #* 2
        norm_dev = 1.0 / self.fnum
        W1 = tf.Variable(tf.random_normal([self.fnum, hidden_num], stddev = norm_dev))
        b1 = tf.Variable(tf.random_normal([hidden_num], stddev = norm_dev))

        W2 = tf.Variable(tf.random_normal([hidden_num, 1], stddev = norm_dev))
        b2 = tf.Variable(tf.random_normal([1], stddev = norm_dev))

        self.model_para = [W1, b1, W2, b2]
        
        # Construct model
        out1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
        out2 = tf.nn.sigmoid(tf.matmul(out1, W2) + b2)

        self.model_y = out2

        # parameters to sovle the model
        self.grad_list = tf.gradients(xs=self.model_para, ys=self.model_y)
        self.para_grad_value = []
        self.update_para = []
        for k in xrange(len(self.model_para)):
            x = tf.placeholder(tf.float32, self.model_para[k].shape)
            self.para_grad_value.append(x)
            self.update_para.append(self.model_para[k].assign_add(self.para_grad_value[k]))

        self.saver = tf.train.Saver()

        self.sess = tf.Session()

        # init variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        #print "init para: ", self.sess.run(self.model_para)

    def load_model(self, model_dir):
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:  
            print >>sys.stderr, "load model failed!"
        #print "load para: ", self.sess.run(self.model_para)

    def save_model(self, model_file):
        self.saver.save(self.sess, model_file)

    def calcNNGrad(self, actions):
        grad_para = []
        for a in actions:
            xs = a.reshape(-1,self.fnum)
            grads = self.sess.run(self.grad_list, feed_dict={self.input_x: xs})
            grad_para.append(grads)

        return grad_para

    def calcGrad(self, a, grad_para, exp_list, status):
        fa = grad_para[a]

        pnum = len(self.model_para)
        ftmp = []
        for i in xrange(pnum):
            ftmp.append(np.zeros(self.model_para[i].shape))

        pro_sum = 0.0
        for i in status.keys():
            fi = grad_para[i]
            pro = np.exp(exp_list[i])
            for k in xrange(pnum):
                ftmp[k] += fi[k] * pro
            pro_sum += pro
        grd = []
        for k in xrange(pnum):
            grd.append(fa[k] - ftmp[k] / pro_sum)
        return grd

    def update(self, doc_list, exp_list, episode, G, gamma, eta):
        M = len(doc_list)
        status = {}
        for i in xrange(M):
            status.setdefault(i)

        actions = []
        for i in xrange(M):
            actions.append(doc_list[i][0])
        grad_para = self.calcNNGrad(actions)

        pnum = len(self.model_para)
        deltaW = []
        for i in xrange(pnum):
            deltaW.append(np.zeros(self.model_para[i].shape))

        for i in xrange(M-1):
            st = episode[i]
            a = st[0]
            grd = self.calcGrad(a, grad_para, exp_list, status)
            for k in xrange(pnum):
                deltaW[k] += math.pow(gamma, i) * G[i] * grd[k]
            del status[a]

        for k in xrange(pnum):
            self.sess.run(self.update_para[k], feed_dict={self.para_grad_value[k]: eta * deltaW[k]})

    def predict(self, doc_list):
        M = len(doc_list)
        actions = []
        for i in xrange(M):
            actions.append(doc_list[i][0])

        ys = self.sess.run(self.model_y, feed_dict={self.input_x: actions})
        scores = np.reshape(ys, [-1])

        return scores

    def Pi(self, exp_list, status):
        M = len(exp_list)

        pi = np.zeros(M)
        curr_sum = 0
        for i in xrange(M):
            if status.has_key(i):
                pro = np.exp(exp_list[i])
                pi[i] = pro
                curr_sum += pro
        pi /= curr_sum
        return pi

if __name__ == "__main__":
    dnn = DnnPolicy(5)
    print "start:"
    actions = np.array([[0.5,0.4,0.7,0.9,0.6], [0.3,0.1,0.2,0.3,0.2], [0.6,0.8,0.8,0.9,0.7]])
    grad_para, Pi = dnn.calcNNGrad(actions)
    print Pi
    print grad_para
    
