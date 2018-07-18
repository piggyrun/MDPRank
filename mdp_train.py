from MDPRank import *

if __name__ == "__main__":
    mdprank = MDPRank("feature_list", gamma = 0.9, eta = 0.02)
    mdprank.load_training("data/train_data", "data/valid_data")
    #mdprank.load_training("data/test_data", "data/valid_data")
    mdprank.train()
    mdprank.save_model("model/dnn_model")
