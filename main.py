import argparse
import numpy as np
import pandas as pd
import json
import time
import datetime
import math
import torch
from utils.minibatch import  AssetDataset
from utils.utilities import *
from models.DySTAGE import DySTAGE

torch.autograd.set_detect_anomaly(True)

# Set random seed for reproducibility
seed = 123
same_seeds(seed)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset and training details
    parser.add_argument('--hist_time_steps', type=int, nargs='?', default=12,
                        help="total time steps used for train, eval and test")
    parser.add_argument('--dataset', type=str, nargs='?', default='asset_15',
                        help='dataset name')
    parser.add_argument('--GPU_ID', type=int, nargs='?', default=0,
                        help='GPU_ID (0/1 etc.)')
    parser.add_argument('--model', type=str, nargs='?', default='DySTAGE',
                        help='model name')
    parser.add_argument('--epochs', type=int, nargs='?', default=300,
                        help='# epochs')
    parser.add_argument('--feat_norm', type=bool, nargs='?', default=True,
                    help='True if normalize feature matrix.')
    parser.add_argument('--adj_norm', type=bool, nargs='?', default=True,
                    help='True if normalize adjacency matrix.')
    parser.add_argument("--early_stop", type=int, default=30,
                        help="patient")
    parser.add_argument("--train_proportion", type=float, default=0.7,
                        help="Proportion for training period")
    parser.add_argument("--valid_proportion", type=float, default=0.15,
                        help="Proportion for validation period")
    # model structure
    parser.add_argument('--residual', type=bool, nargs='?', default=True,
                        help='Use residual')
    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--temporal_drop', type=float, nargs='?', default=0.5,
                        help='Temporal attention Dropout (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0005,
                        help='Initial learning rate for self-attention model.')
    parser.add_argument('--node_dim', type=int, nargs='?', default=8,
                        help='Encoder layer config: # units')
    parser.add_argument('--n_heads', type=int, nargs='?', default=16,
                        help='Encoder layer config: # attention heads')  
    parser.add_argument('--attention_layers', type=int, nargs='?', default=1,
                        help='Encoder layer config: # attention heads')    
    parser.add_argument('--centrality', type=bool, nargs='?', default=False,
                        help='centrality encoding')
    parser.add_argument('--spatial', type=bool, nargs='?', default=True,
                        help='spatial encoding')  
    parser.add_argument('--edge', type=bool, nargs='?', default=True,
                        help='edge encoding')    
    parser.add_argument('--temporal_head_config', type=str, nargs='?', default='16',
                        help='Encoder layer config: # attention heads in each Temporal layer')
    parser.add_argument('--temporal_layer_config', type=str, nargs='?', default='128',
                        help='Encoder layer config: # units in each Temporal layer')
    args = parser.parse_args()
    print(args)

    # set up results directory
    now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')
    RESULTS_DIR = "results/"+now
    check_directories([RESULTS_DIR])

    # save experiment configuration
    with open(RESULTS_DIR+'/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('Results directory:', RESULTS_DIR)

    # load dataset
    adjs = load_graphs(args.dataset)
    all_feats = load_features(args.dataset)
    edge_feat = load_edge_feat(args.dataset)
    shortest_paths = load_shortest_paths(args.dataset)
    if args.dataset == 'sp':
        # daily data uses previous 3 months to build graphs
        feats = all_feats[61:]
        valid_feat_idx = 4 # Feature index to indicate whether asset exists
    else:
        # monthly data uses previous 36 months to build graphs
        feats = all_feats[36:] 
        valid_feat_idx = 1

    # Extract feature and node dimensions
    feat_dim = feats[0].shape[1]
    num_nodes = adjs[0].shape[0]
    edge_scale = edge_feat[0].shape[-1]

    print('Total time steps:', len(adjs))
    print('Total number of assets:', num_nodes)
    print('Total number of features:', feat_dim)

    # Ensure the number of historical time steps is valid
    assert args.hist_time_steps < len(adjs), "Time steps is illegal"

    # Set device to GPU if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('train model on gpu')
    else:
        device = 'cpu'
        print('train model on cpu')

    # Initialize dataloader and model
    dataloader = AssetDataset(args, feats, adjs, shortest_paths, edge_feat)
    
    if args.model == 'DySTAGE':
        model = DySTAGE(args, num_nodes, feat_dim, edge_scale, valid_feat_idx).to(device)
    else:
        print('Model does not exist!')
        exit()

    # training loop
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_epoch_val, best_epoch = 1000000, 0
    patient = 0
    t_total0 = time.time()
    time_list = []
    mem_list = []

    for epoch in range(args.epochs):
        # training phase
        t0 = time.time()
        model.train()
        epoch_loss = []

        for idx, train_data in enumerate(dataloader.train):
            train = to_device(train_data, device)
            opt.zero_grad()
            loss, _, _ = model.get_loss(train)
            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())
            
        # Validation phase
        model.eval()
        eval_perf = []
        for idx, valid_data in enumerate(dataloader.valid):
            valid = to_device(valid_data, device)
            eval_loss_idx, eval_pred, eval_labels = model.get_loss(valid)
            perf_idx = calculate_perf(eval_pred, eval_labels)
            eval_perf_epoch = [eval_loss_idx.detach().cpu().numpy(), math.sqrt(eval_loss_idx.detach().cpu().numpy())]
            eval_perf_epoch.extend(perf_idx)
            eval_perf.append(eval_perf_epoch)
        eval_perf = np.array(eval_perf)
        
        # save best model regarding minimal valid loss
        if eval_perf[:,0].mean() < best_epoch_val:
            best_epoch_val = eval_perf[:,0].mean()
            best_epoch = epoch
            torch.save(model.state_dict(), RESULTS_DIR+"/model.pt")
            patient = 0        
        else:
            patient += 1
            if patient > args.early_stop:
                break
        
        # Track time and memory usage
        gpu_mem_alloc = torch.cuda.max_memory_allocated(device) / 1000000 if torch.cuda.is_available() else 0
        time_list.append(time.time() - t0)
        mem_list.append(gpu_mem_alloc)

        print("Epoch {:<3}, Time: {:.3f}, GPU: {:.1f}MiB, Train Loss = {:.4f}, Valid Loss = {:.4f}".format(\
                epoch, time.time() - t0, gpu_mem_alloc, np.mean(epoch_loss), eval_perf[:,0].mean()))
                
        
    # Load and test best model
    model.load_state_dict(torch.load(RESULTS_DIR+"/model.pt"))
    model.eval()
    test_perf, test_preds, test_labels = [], [], []

    for idx, test_data in enumerate(dataloader.test):
        test = to_device(test_data, device)
        test_loss_idx, test_pred_idx, test_labels_idx = model.get_loss(test)
        perf_idx = calculate_perf(test_pred_idx, test_labels_idx)
        test_perf_epoch = [test_loss_idx.detach().cpu().numpy(), math.sqrt(test_loss_idx.detach().cpu().numpy())]
        test_perf_epoch.extend(perf_idx)
        test_perf.append(test_perf_epoch)
        test_preds.append(test_pred_idx)
        test_labels.append(test_labels_idx)
    test_perf = np.array(test_perf)
    port_perf = portfolio_perf(test_preds, test_labels, args)

    print("Best Epoch {:<3}, Test MSE = {:.4f}, Test RMSE = {:.4f}, Test MAE = {:.4f}, Test MAPE = {:.4f}, CR = {:.4f}, AR = {:.4f}, SR = {:.4f}".format(\
            best_epoch, test_perf[:,0].mean(), test_perf[:,1].mean(), test_perf[:,2].mean(), test_perf[:,3].mean(),port_perf[0],port_perf[1],port_perf[2],))
                    
    # Save predictions and labels
    pred = pd.DataFrame(test_preds)
    pred.to_csv(RESULTS_DIR+'/test_pred.csv', index=False, header=False)
    labels = pd.DataFrame(test_labels)
    labels.to_csv(RESULTS_DIR+'/test_label.csv', index=False, header=False)
