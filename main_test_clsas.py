import os
import time
import torch
import argparse

from model import SASRec
from utils import *
from modules import NCELoss
import torch.nn as nn

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--rec_weight', default=1.0, type=float)
parser.add_argument("--cl_weight", default=0.1, type=float)
parser.add_argument("--temp", default=0.6, type=float)
parser.add_argument("--topk", default=10, type=float)



args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()


def info_nec_loss(seq_1, seq_2, temperature, batch_size, sim="dot"):
    N = 2 * batch_size

    z = torch.cat((seq_1, seq_2), dim=0)

    if sim == "cos":
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temperature

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)


    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0

    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)

    return logits, labels

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args.dataset)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset

    print(usernum)
    print(itemnum)
    num_batch = len(user_train) // args.batch_size  # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb;

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args.topk, args)
        print(f'test (NDCG@{args.topk}: %.4f, HR@{args.topk}: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    # --- cl learning criterion --- #
    nce_criterion = NCELoss(1.0, args.device)
    nce_fct = nn.CrossEntropyLoss()


    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # (u, seq, pos, neg), (cl_u, cl_seq, cl_pos, cl_neg), (cl2_u, cl2_seq, cl2_pos, cl2_neg) = sampler.next_batch()



    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            (u, seq, pos, neg), (cl_u, cl_seq, cl_pos, cl_neg), (cl2_u, cl2_seq, cl2_pos, cl2_neg) = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            indices = np.where(pos != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            # --- cl_learning --- #
            cl_u, cl_seq, cl_pos, cl_neg = np.array(cl_u), np.array(cl_seq), np.array(cl_pos), np.array(cl_neg)
            cl2_u, cl2_seq, cl2_pos, cl2_neg = np.array(cl2_u), np.array(cl2_seq), np.array(cl2_pos), np.array(cl2_neg)


            feat1 = model.log2feats(cl_seq)
            feat2 = model.log2feats(cl2_seq)

            feat1_flatten = feat1.view(feat1.shape[0], -1)
            feat2_flatten = feat2.view(feat2.shape[0], -1)
            # cl_loss = nce_criterion(feat1_flatten, feat2_flatten)
            nce_logits, nce_labels = info_nec_loss(feat1_flatten, feat2_flatten, args.temp, feat1_flatten.shape[0])
            nce_loss = nce_fct(nce_logits, nce_labels)

            joint_loss = args.rec_weight * loss + args.cl_weight * nce_loss

            for param in model.item_emb.parameters(): joint_loss += args.l2_emb * torch.norm(param)

            adam_optimizer.zero_grad()
            joint_loss.backward()
            adam_optimizer.step()
            print("loss in epoch {} iteration {}: {}".format(epoch, step, joint_loss.item()))  # expected 0.4~0.6 after init few epochs

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, 10, args)
            t_valid = evaluate_valid(model, dataset, 10, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'CLS_SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.batch_size={}_temp={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen, args.batch_size, args.temp)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    sampler.close()
    print("Done")
