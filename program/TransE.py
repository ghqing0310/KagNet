# 这份代码是根据openKE的指引改编
# 其实没有很多改动，注意输入输出就好了
from OpenKE.config import Config
from OpenKE import models
import numpy as np
import tensorflow as tf
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('opt_method', help='SGD/Adagrad/...')
parser.add_argument('pretrain', help='0/1', type=int, default=1)
args = parser.parse_args()


def run():

    opt_method = args.opt_method
    int_pretrain = args.pretrain
    if int_pretrain == 1:
        pretrain = True
    elif int_pretrain == 0:
        pretrain = False
    else:
        raise ValueError('arg "pretrain" must be 0 or 1')

    config = Config()
    config.set_in_path("../dataset/")
    config.set_log_on(1)  # set to 1 to print the loss

    config.set_work_threads(30)
    config.set_train_times(10000)  # number of iterations
    config.set_nbatches(512)  # batch size
    config.set_alpha(0.001)  # learning rate

    config.set_bern(0)
    config.set_dimension(100)
    config.set_margin(1.0)
    config.set_ent_neg_rate(1)
    config.set_rel_neg_rate(0)
    config.set_opt_method(opt_method)

    '''revision starts'''
    config.set_pretrain(pretrain)

    OUTPUT_PATH = "../dataset/emb_init/"

    '''revision ends'''

    # Model parameters will be exported via torch.save() automatically.
    config.set_export_files(OUTPUT_PATH + 'transe.' + opt_method + '.tf', steps=500)
    # Model parameters will be exported to json files automatically.
    config.set_out_files(OUTPUT_PATH + "transe."+opt_method+".vec.json")

    print("Opt-method: %s" % opt_method)
    print("Pretrain: %d" % pretrain)
    config.init()
    config.set_model(models.TransE)

    print("Begin training TransE")

    config.run()


if __name__ == "__main__":
    run()
