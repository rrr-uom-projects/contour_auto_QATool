## utils.py
# some useful functions!
import numpy as np
from itertools import cycle
import torch
import shutil
import os
import logging
import argparse
import sys
import math
import warnings
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def getDirs(parent_dir):
    ls = []
    for dir_name in os.listdir(parent_dir):
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path):
            ls.append(dir_name)
    return ls

def windowLevelNormalize(image, level, window):
    minval = level - window/2
    maxval = level + window/2
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / window)
    return wld

def try_mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except OSError:
        pass

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

class RunningAverage:
    # Computes and stores the average
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count

class ConfusionMatrix:
    # Computes and plots a confusion matrix for the classification task
    def __init__(self, n_classes=5):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes), int)

    def update(self, targets, soft_preds):
        hard_preds = np.argmax(soft_preds, axis=1)
        self.confusion_matrix += confusion_matrix(y_true=targets, y_pred=hard_preds, labels=np.arange(self.n_classes)).astype(int)

    def _normalise_by_true(self):
        return np.transpose(np.transpose(self.confusion_matrix) / self.confusion_matrix.sum(axis=1))

    def _normalise_by_pred(self):
        return self.confusion_matrix / self.confusion_matrix.sum(axis=0)

    def gen_matrix_fig(self):
        # setup
        fig, ax = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
        # normalise by true classes
        normed_confusion_matrix = self._normalise_by_true()
        I_blanker = np.ones((self.n_classes, self.n_classes))
        I_blanker[np.identity(self.n_classes, bool)] = np.nan
        ax.imshow(normed_confusion_matrix, cmap='Greens', vmin=0, vmax=1)
        ax.imshow(normed_confusion_matrix*I_blanker, cmap='Reds', vmin=0, vmax=1)
        for target_idx in range(self.n_classes):
            for pred_idx in range(self.n_classes):
                ax.text(pred_idx, target_idx, s=f"{np.round(normed_confusion_matrix[target_idx, pred_idx]*100, 1)}%\nn={self.confusion_matrix[target_idx, pred_idx]}", ha='center', va='center')
        ax.set_xlabel("Pred class")
        ax.set_ylabel("True class")
        ax.set_xticks(np.arange(self.n_classes))
        ax.set_yticks(np.arange(self.n_classes))
        if self.n_classes == 5:
            ax.set_xticklabels(["<-2.5mm","-2.5 - -1mm","-1 - 1mm", "1 - 2.5mm", ">2.5mm"])
            ax.set_yticklabels(["<-2.5mm","-2.5 - -1mm","-1 - 1mm", "1 - 2.5mm", ">2.5mm"])
        elif self.n_classes == 3:
            ax.set_xticklabels(["<-2.5mm","-2.5 -> 2.5mm", ">2.5mm"])
            ax.set_yticklabels(["<-2.5mm","-2.5 -> 2.5mm", ">2.5mm"])
        else:
            raise NotImplementedError("Check confusion matrix labels!")
            exit()
        # return figure
        return fig

    def retrieve_data(self):
        return self.confusion_matrix

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def k_fold_split_train_val_test(dataset_size, fold_num, seed):
    k = int(fold_num-1)
    train_ims, val_ims, test_ims = math.floor(dataset_size*0.7), math.floor(dataset_size*0.1), math.ceil(dataset_size*0.2)
    if dataset_size - (train_ims+val_ims+test_ims) == 1:
        val_ims += 1 # put the extra into val set
    try:
        assert(train_ims+val_ims+test_ims == dataset_size)
    except AssertionError:
        print("Check the k fold data splitting, something's dodgy...")
        exit(1)
    train_inds, val_inds, test_inds = [], [], []
    # initial shuffle
    np.random.seed(seed)
    shuffled_ind_list = np.random.permutation(dataset_size)
    # allocate dataset indices based upon the fold number --> not the prettiest or most efficient implementation, but functional
    cyclic_ind_list = cycle(shuffled_ind_list)
    for i in range(k*test_ims):
        next(cyclic_ind_list)   # shift start pos
    for i in range(test_ims):
        test_inds.append(next(cyclic_ind_list))
    for i in range(train_ims):
        train_inds.append(next(cyclic_ind_list))
    for i in range(val_ims):
        val_inds.append(next(cyclic_ind_list))
    return train_inds, val_inds, test_inds

def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)
    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)

def get_checkpoint_dir(root_dir, args):
    checkpoint_dir = root_dir
    len_orig = len(checkpoint_dir)
    # LR
    if args.init_lr == 0.01:
        checkpoint_dir = checkpoint_dir + "lr1e2"
    elif args.init_lr == 0.005:
        checkpoint_dir = checkpoint_dir + "lr5e3"
    elif args.init_lr == 0.001:
        checkpoint_dir = checkpoint_dir + "lr1e3"
    else:
        print("Please add init_lr to checkpoint_dir tree function")
        exit()
    if args.lr_sched == "noS":
        checkpoint_dir = checkpoint_dir + "_noS"
    elif args.lr_sched == "expS":
        checkpoint_dir = checkpoint_dir + "_expS"
    elif args.lr_sched == "cosS":
        checkpoint_dir = checkpoint_dir + "_cosS"
    # batch size
    checkpoint_dir = checkpoint_dir + f"_bs{args.bs}"
    # deformations to sample per epoch
    if args.dtspe != 25:
        checkpoint_dir = checkpoint_dir + f"_dtspe{args.dtspe}"
    # processor type
    if args.processor == "GAT":
        checkpoint_dir = checkpoint_dir + "_GAT"
    elif args.processor == "deeper_spline":
        checkpoint_dir = checkpoint_dir + "_deepSpl"
    elif args.processor == "wider_spline":
        checkpoint_dir = checkpoint_dir + "_wideSpl"
    elif args.processor == "better_GAT":
        checkpoint_dir = checkpoint_dir + "_betGAT"
    elif args.processor == "simple_spline":
        checkpoint_dir = checkpoint_dir + "_simSpl"
    elif args.processor == "simple_spline128":
        checkpoint_dir = checkpoint_dir + "_simSpl128"
    # decoder type
    if args.decoder_feat != 128:
        checkpoint_dir = checkpoint_dir + f"_dec{args.decoder_feat}"
    if args.decoder_bn == False:
        checkpoint_dir = checkpoint_dir + "_decNoBN"
    # dropout 
    if args.dropout == False:
        checkpoint_dir = checkpoint_dir + "_noDrop"
    # spline bits
    if args.spline_deg != 1:
        checkpoint_dir = checkpoint_dir + f"_deg{args.spline_deg}"
    if args.kernel_size != 5:
        checkpoint_dir = checkpoint_dir + f"_ks{args.kernel_size}"
    if args.aggr != "mean":
        checkpoint_dir = checkpoint_dir + f"_aggr{args.aggr}"
    # pretrained bit
    if args.lock_pretrained_CNN:
        checkpoint_dir = checkpoint_dir + "_cnnLock"
    if args.encAbl:
        checkpoint_dir = checkpoint_dir + "_blankCTAbl"
    if args.GNNAbl:
        checkpoint_dir = checkpoint_dir + "_GNNAbl"
    if args.preAbl:
        checkpoint_dir = checkpoint_dir + "_noPre"
    if args.loBeta:
        checkpoint_dir = checkpoint_dir + "_loBeta"
    # finally get full experiment type
    exp_type = checkpoint_dir[len_orig:]
    return checkpoint_dir, exp_type