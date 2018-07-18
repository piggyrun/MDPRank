from MDPRank import *

if __name__ == "__main__":
    #mdprank = MDPRank("feature_list.nor")
    #mdprank = MDPRank("feature_list.mul")
    mdprank = MDPRank("feature_list")
    #mdprank.load_model("svm_model.nor")
    #mdprank.load_model("svm_model.mul")
    #mdprank.load_model("mdp_model")
    mdprank.load_model("model")
    print "start testing"
    mdprank.load_testing("data/test_data")
    mdprank.test()
