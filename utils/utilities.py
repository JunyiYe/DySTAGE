import pickle as pkl
import copy, os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import math
from pyfolio import timeseries
import pandas as pd
import torch
import random

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_graphs(dataset_str):
    with open("data/{}/{}".format(dataset_str, "graph.pkl"), "rb") as f:
        graphs = pkl.load(f)
    print("Loaded {} graphs ".format(len(graphs)))
    return graphs

def load_features(dataset_str):
    with open(f"data/{dataset_str}/features.pkl", "rb") as f:
        features = pkl.load(f)
    return features

def load_edge_feat(dataset_str):
    with open(f"data/{dataset_str}/edge_feat.pkl", "rb") as f:
        edge_feat = pkl.load(f)
    return edge_feat

def load_shortest_paths(dataset_str):
    shortest_paths = torch.load(f"data/{dataset_str}/shortest_paths.pt") 
    return shortest_paths


def check_directories(directories):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

def to_device(batch, device):
    data = copy.deepcopy(batch)
    idx, graphs, labels = data.values()
    data["idx"] = idx[0].to(device)
    data["hist_graphs"] = [x.to(device) for x in graphs]
    data["labels"] = labels[0].to(device)
    return data

def mape_clip(label, pred, threshold = 0.1):
    v = np.clip(np.abs(label), threshold, None)
    diff = np.abs((label - pred) / v)
    return 100.0 * np.mean(diff)

def calculate_perf(pred, label):
    mae = mean_absolute_error(label, pred)
    mape = mape_clip(label, pred)
    rmse = math.sqrt(mean_squared_error(label, pred))
    return [mae, mape, rmse]

def init_path(path, args):
    if not os.path.exists(path + "/train_log.txt"):
        os.mknod(path + "/train_log.txt")
    if not os.path.exists(path + "/{}_result.txt".format(args.dataset)):
        os.mknod(path + "/{}_result.txt".format(args.dataset))


def perf_sp(port):
    sharpe = timeseries.sharpe_ratio(port,period='daily')
    res = (port+1).cumprod()-1
    ret = res[-1]
    aret = timeseries.annual_return(port, period='daily')
    return [100*ret, 100*aret, sharpe]
    
def perf(port):
    sharpe = timeseries.sharpe_ratio(port,period='monthly')
    res = (port+1).cumprod()-1
    ret = res[-1]
    aret = timeseries.annual_return(port, period='monthly')
    return [100*ret, 100*aret, sharpe]


def portfolio_perf(pred, labels, args):

    pred = pd.DataFrame(pred)
    labels = pd.DataFrame(labels)

    label_long_port = np.empty(labels.shape[0])
    our_long_port = np.empty(labels.shape[0])

    for i in range(len(labels)):
        k = int(len(pred.iloc[i].dropna())*0.1)
        our_t_f = np.array(pred.iloc[i].dropna()).argsort()[-k:]
        label_long_port[i] = np.array(labels.iloc[i].dropna()).mean()
        our_long_port[i] = np.array(labels.iloc[i].dropna())[our_t_f].mean()

    if args.dataset == 'sp':
        res = perf_sp(our_long_port)
    else:
        res = perf(our_long_port)
        
    return res
    