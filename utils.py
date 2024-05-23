import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from dataAug import Random, Reorder, Crop, Mask, Substitute, Insert
from modules import OnlineItemSimilarity

random_seq = Random()
reorder_seq = Reorder(beta=0.2)
crop_seq = Crop(tao=0.2)
mask_seq = Mask(gamma=0.4)


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, cl_queue, cl2_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        # ---- cl_dataset ---- #

        switch = random.sample(range(3), k=2)

        if switch[0] == 0:
            aug1_user_train = crop_seq(user_train[user])
        elif switch[0] == 1:
            aug1_user_train = mask_seq(user_train[user])
        elif switch[0] == 2:
            aug1_user_train = reorder_seq(user_train[user])

        if switch[1] == 0:
            aug2_user_train = crop_seq(user_train[user])
        elif switch[1] == 1:
            aug2_user_train = mask_seq(user_train[user])
        elif switch[1] == 2:
            aug2_user_train = reorder_seq(user_train[user])

        # aug1_user_train = reorder_seq(user_train[user])
        # aug2_user_train = mask_seq(user_train[user])

        ts1 = set(aug1_user_train)
        ts2 = set(aug2_user_train)

        seq1 = np.zeros([maxlen], dtype=np.int32)
        pos1 = np.zeros([maxlen], dtype=np.int32)
        neg1 = np.zeros([maxlen], dtype=np.int32)
        nxt1 = aug1_user_train[-1]
        idx1 = maxlen - 1

        seq2 = np.zeros([maxlen], dtype=np.int32)
        pos2 = np.zeros([maxlen], dtype=np.int32)
        neg2 = np.zeros([maxlen], dtype=np.int32)
        nxt2 = aug2_user_train[-1]
        idx2 = maxlen - 1

        for i in reversed(aug1_user_train[:-1]):
            seq1[idx1] = i
            pos1[idx1] = nxt1
            if nxt1 != 0: neg1[idx1] = random_neq(1, itemnum + 1, ts1)
            nxt1 = i
            idx1 -= 1
            if idx1 == -1: break

        for i in reversed(aug2_user_train[:-1]):
            seq2[idx2] = i
            pos2[idx2] = nxt2
            if nxt2 != 0: neg2[idx2] = random_neq(1, itemnum + 1, ts2)
            nxt2 = i
            idx2 -= 1
            if idx2 == -1: break

        return (user, seq, pos, neg), (user, seq1, pos1, neg1), (user, seq2, pos2, neg2)

    np.random.seed(SEED)
    while True:
        one_batch = []
        cl_batch = []
        cl2_batch = []
        for i in range(batch_size):
            s, cl_s, cl2_s = sample()
            one_batch.append(s)
            cl_batch.append(cl_s)
            cl2_batch.append(cl2_s)

        result_queue.put(zip(*one_batch))
        cl_queue.put(zip(*cl_batch))
        cl2_queue.put(zip(*cl2_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.cl_queue = Queue(maxsize=n_workers * 10)
        self.cl2_queue = Queue(maxsize=n_workers * 10)

        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      self.cl_queue,
                                                      self.cl2_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get(), self.cl_queue.get(), self.cl2_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, topk, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        # print("predictions: ", predictions.size())
        predictions = predictions[0]  # - for 1st argsort DESC
        # print("predictions: ", predictions.size())
        rank = predictions.argsort().argsort()[0].item()
        # print("rank: ", rank)
        valid_user += 1

        if rank < topk:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, topk, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
