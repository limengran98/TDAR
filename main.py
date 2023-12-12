import argparse
import io
import json
import os
import time
import csv

import numpy as np
import pandas as pd

import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

from data import load_data, is_continuous, is_large, to_edge_tensor
from models.tdar import TDAR
from models.fp import APA
from utils import *

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,SAGEConv, GATConv,ChebConv,GINConv,SGConv

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,f1_score, normalized_mutual_info_score, adjusted_rand_score, ndcg_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")



def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cora')
    parser.add_argument('--y-ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--sampling', type=str2bool, default=False)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--out', type=str, default='../out')

    parser.add_argument('--lr', type=float, default=1e-3) #photo computers 1e-2; cora citeseer 1e-3 
    parser.add_argument('--lamda', type=float, default=0.1)
    parser.add_argument('--lambda_penalty', type=float, default=0.1)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--num_iter', type=float, default=30)
    parser.add_argument('--threshold', type=float, default=0.2)

    parser.add_argument('--emb-norm', type=str, default='unit')
    parser.add_argument('--dec-bias', type=str2bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.8)#photo computers 0.2; cora citeseer 0.8

    parser.add_argument('--layers', type=int, default=2) #photo computers 1; cora citeseer 2 
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=2000) #photo computers 400; cora citeseer 2000 
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--updates', type=int, default=10)
    parser.add_argument('--conv', type=str, default='lin') #photo computers gcn; cora citeseer lin
    parser.add_argument('--x-loss', type=str, default='base')
    parser.add_argument('--x-type', type=str, default='fp')
    parser.add_argument('--model', type=str, default='ours')
    return parser.parse_args()


def main():
    """
    Main function.
    """
    args = parse_args()
    print(args)

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    edge_index, x_all, y_all, trn_nodes, val_nodes, test_nodes = load_data(
        args.data, split=(0.4, 0.1, 0.5), seed=args.seed) #0.8-0.2

    num_nodes = x_all.size(0)
    num_features = x_all.size(1)
    num_classes = (y_all.max() + 1).item()

    x_nodes = trn_nodes
    missing_nodes = test_nodes
    x_features = x_all[trn_nodes]
    fp_features = x_all.clone()
    fp_features[missing_nodes] = 0
    y_nodes, y_labels = trn_nodes, y_all[trn_nodes]


    model = TDAR(
            edge_index, num_nodes, num_features, num_classes, args.hidden_size, args.eps, args.alpha, args.num_iter,
            args.lamda, args.lambda_penalty, args.threshold, args.layers, args.conv, args.dropout,
            args.x_type, args.x_loss, args.emb_norm, trn_nodes, missing_nodes, x_features,fp_features,
            args.dec_bias)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    device = to_device(args.gpu)
    edge_index = edge_index.to(device)
    model = model.to(device)
    trn_nodes = trn_nodes.to(device)
    x_all = x_all.to(device)
    x_features = x_features.to(device)
    y_all = y_all.to(device)

    if y_nodes is not None:
        y_nodes = y_nodes.to(device)
        y_labels = y_labels.to(device)

    def update_model(epoch, step):
        model.train()
        losses = model.to_losses(edge_index, trn_nodes, x_features, y_nodes, y_labels, args.model, epoch)
        if step:
            optimizer.zero_grad()
            sum(losses).backward()
            optimizer.step()
        return tuple(l.item() for l in losses)

    @torch.no_grad()
    def evaluate_model():
        model.eval()
        z, x_hat_, y_hat = model(edge_index)
        out_list = []
        for nodes in [x_nodes, val_nodes, test_nodes]:
            if is_continuous(args.data):
                score = to_rmse(x_hat_[nodes], x_all[nodes])
            else:
                score = to_recall(x_hat_[nodes], x_all[nodes], k=10)
            out_list.append(score)
        return out_list

    def is_better(curr_acc_, best_acc_):
        if is_continuous(args.data):
            return curr_acc_ <= best_acc_
        else:
            return curr_acc_ >= best_acc_

    if not args.silent:
        print('-' * 47)
        print('epoch x_loss y_loss r_loss  trn    val    test')

    logs = []
    saved_model, best_epoch, best_result = io.BytesIO(), 0, []
    best_acc = np.inf if is_continuous(args.data) else 0
    for epoch in range(args.epochs + 1):
        loss_list = []
        for _ in range(args.updates):
            loss_list = update_model(epoch, epoch > 0)
        acc_list = evaluate_model()
        curr_result = [epoch, loss_list, acc_list]

        val_acc = acc_list[2]
        if is_better(val_acc, best_acc):
            saved_model.seek(0)
            torch.save(model.state_dict(), saved_model)
            best_epoch = epoch
            best_acc = val_acc
            best_result = curr_result

        if not args.silent and epoch % (args.epochs // min(args.epochs, 20)) == 0:
            print_log(*curr_result)

        if args.patience > 0 and epoch >= best_epoch + args.patience:
            break

    saved_model.seek(0)
    model.load_state_dict(torch.load(saved_model))

    val_res, z, x_hat, y_hat = evaluate_last(args.data, model, edge_index, val_nodes, x_all)
    test_res, z, x_hat, y_hat = evaluate_last(args.data, model, edge_index, test_nodes, x_all)

    out_best = dict(epoch=best_epoch, val_res=val_res, test_res=test_res)
    if not args.silent:
        print('-' * 47)
        print_log(*best_result)
        print('-' * 47)
        print(json.dumps(out_best, indent=4, sort_keys=True))
    else:
        out = {arg: getattr(args, arg) for arg in vars(args)}
        out['out'] = out_best
        print(json.dumps(out))
    print(out_best['test_res'])


## clustering
    kmeans = KMeans(n_clusters=num_classes)
    y_pred = kmeans.fit_predict(z.cpu().numpy())
    y_pred = torch.Tensor(y_pred).long()  
    y_true = y_all.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    nmi = normalized_mutual_info_score(y_true, y_pred_np)
    ac, f1_ = cluster_acc(y_true, y_pred_np)
    ari = adjusted_rand_score(y_true, y_pred_np)
    print(f'AC: {ac:.4f}, '
        f'NMI: {nmi:.4f}, '
        f'ARI: {ari:.4f}, '
        f'F1: {f1_:.4f}')

## classification
    x_hat = x_hat[missing_nodes].to(device)
    y_all = y_all[missing_nodes]

    edge_index, _ = subgraph(sorted_nodes.to(device), edge_index, relabel_nodes=True)
    edge_index = edge_index.to(device)

    def train(feature, edge_index, y_all, train_idx):
        cl_model.train()
        optimizer.zero_grad()
        out = cl_model(feature, edge_index, None)
        loss = F.cross_entropy(out[train_idx], y_all[train_idx])
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test(feature, edge_index, y_all, test_idx):

        cl_model.eval()
        pred = cl_model(feature, edge_index, None)
        mask = test_idx
        f1 = f1_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        acc = accuracy_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy())
        pre = precision_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        rec = recall_score(y_all[mask].cpu().numpy(),torch.argmax(pred[mask],dim=1).cpu().numpy(),average = 'macro')
        return acc, f1, pre, rec

    node_Idx = shuffle(np.arange(x_hat.shape[0]), random_state=72)
    KF = KFold(n_splits=5)
    split_data = KF.split(node_Idx)
    acc_list = []
    f1_list = []
    pre_list = []
    rec_list = []
    for i in range(1):
        for train_idx, test_idx in split_data:
            cl_model = GCN(x_hat.shape[1], 256, num_classes).to(device) 
            optimizer = torch.optim.Adam(cl_model.parameters(),lr=args.lr) 
            best_acc = best_f1 = best_pre = best_rec = 0
            for epoch in range(0, 1001):
                loss = train(x_hat, edge_index, y_all,train_idx)
                acc, f1, pre, rec = test(x_hat, edge_index, y_all, test_idx)
                if acc > best_acc:
                    best_acc = acc
                if f1 > best_f1:
                    best_f1 = f1
                if pre > best_pre:
                    best_pre = pre
                if rec > best_rec:
                    best_rec= rec
            acc_list.append(best_acc)
            f1_list.append(best_f1)
            pre_list.append(best_pre)
            rec_list.append(best_rec)
        print("-----------------") 
        print('Mean, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(np.mean(pre_list), np.mean(rec_list),np.mean(f1_list), np.mean(acc_list)))
        print('Std, \n  Macro_Pre: {}, \n Macro_Rec: {}, \n Macro_F1: {}, \n Acc: {}'.format(np.var(pre_list)*100, np.var(rec_list)*100,np.var(f1_list)*100, np.var(acc_list)*100))


    if args.save:
        results = []
        a_values = lamda
        b_values = lambda_penalty
        recalls_10 = out_best['test_res'][0]
        recalls_20 = out_best['test_res'][1]
        recalls_50 = out_best['test_res'][2]
        ndcg_10 = out_best['test_res'][3]
        ndcg_20 = out_best['test_res'][4]
        ndcg_50 = out_best['test_res'][5]
        acc = np.mean(acc_list)
        ac = ac
        nmi = nmi
        ari = ari
        f1_ = f1_
        result = [a_values, b_values, recalls_10, recalls_20, recalls_50, ndcg_10, ndcg_20, ndcg_50, acc, ac, nmi, ari, f1_]
        with open(args.data+'results_esp.csv', 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            #header = ["a", "b", "Recall@10", "Recall@20", "Recall@50", "NDCG@10", "NDCG@20", "NDCG@50", "ACC", "AC", "NMI", "ARI", "F1"]
            #csv_writer.writerow(header)
            csv_writer.writerow(result)



if __name__ == '__main__':
    start = time.time()
    # for lamda in [0.01,0.05,0.1,0.5,1,5,10]:
    #     for lambda_penalty in [0.01,0.05,0.1,0.5,1,5,10]:
    #         main()
    # lamda= 0.05
    # lambda_penalty = 0.1
    # eps=0.01
    # alpha=0.9
    # num_iter = 30
    #for num_iter in [1,5,10,20,30,50,100,200]:
    #for eps in [10,1,0.1,0.01,0.001]:
    #for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    main()
    end = time.time()
    print(end-start)
