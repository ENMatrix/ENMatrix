import torch
import random
import glob
import inspect
import argparse
import copy
import pdb

import numpy as np
import pandas as pd
import os.path as osp

from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn import preprocessing

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet

parser = argparse.ArgumentParser()
parser.add_argument('--uci_data', type=str, default='en_matrix')
parser.add_argument('--model_folder_name', type=str, default='Provide model path')
parser.add_argument('--csv_data_file_name', type=str, default='Provide CSV data file name path') 
parser.add_argument('--min_proba', type=float, default=1e-8) 
args = parser.parse_args()
print(args)

UCI_DATA = args.uci_data
FOLDER_PATH = f"uci/test/{UCI_DATA}/{args.model_folder_name}/"

def create_node(df, mode):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1]*ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1: # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+1))
        feature_node[np.arange(ncol), feature_ind+1] = 1
        sample_node = np.zeros((nrow,ncol+1))
        sample_node[:,0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node

def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col # obj
        edge_end = edge_end + list(n_row+np.arange(n_col)) # att    
    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start
    return (edge_start_new, edge_end_new)

def create_edge_attr(df):
    nrow, ncol = df.shape
    edge_attr = []
    for i in range(nrow):
        for j in range(ncol):
            edge_attr.append([float(df.iloc[i,j])])
    edge_attr = edge_attr + edge_attr
    return edge_attr

uci_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
df_np = np.loadtxt(f'uci/raw_data/{UCI_DATA}/data/data.txt')
df_y = pd.DataFrame(df_np[:, -1:])
print(df_y.head())
df_X = pd.DataFrame(df_np[:, :-1])
print(df_X.head())

if len(df_y.shape)==1:
    df_y = df_y.to_numpy()
elif len(df_y.shape)==2:
    df_y = df_y[0].to_numpy()

normalize = True

if normalize:
    x = df_X.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_X = pd.DataFrame(x_scaled)
    print(df_X.head())

edge_start, edge_end = create_edge(df_X)
edge_index = torch.tensor([edge_start, edge_end], dtype=int).to('cpu')
edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float).to('cpu')
node_init = create_node(df_X, 0) 
x = torch.tensor(node_init, dtype=torch.float).to('cpu')
y = torch.tensor(df_y, dtype=torch.float).to('cpu')
model = torch.load(FOLDER_PATH + 'model.pt').to('cpu')
impute_model = torch.load(FOLDER_PATH + 'impute_model.pt').to('cpu')
model.eval()
impute_model.eval()
with torch.no_grad():
    print(x.size())
    print(edge_attr.size())
    print(edge_index.size())
    x_embd = model(x, edge_attr, edge_index)
    pred = impute_model([x_embd[edge_index[0], :], x_embd[edge_index[1], :]])
    vals = edge_attr.squeeze().numpy().reshape((edge_attr.size()[0] // (len(list(df_X.columns)))), (len(list(df_X.columns))))
    preds = pred.squeeze().numpy().reshape((edge_attr.size()[0] // ((len(list(df_X.columns))))), (len(list(df_X.columns))))
    print(vals.shape, preds.shape)

    scale_factor = np.max(np.array(np.loadtxt(f"uci/raw_data/{UCI_DATA}/data/data.txt"))[:, :-1], axis=0) \
        - np.min(np.array(np.loadtxt(f"uci/raw_data/{UCI_DATA}/data/data.txt"))[:, :-1], axis=0)
    scale_min = np.min(np.array(np.loadtxt(f"uci/raw_data/{UCI_DATA}/data/data.txt"))[:, :-1], axis=0)
    actual_preds = preds[:(edge_attr.size()[0] // ((len(list(df_X.columns))) * 2)), :] * scale_factor + scale_min

    actual_vals = np.round(vals[:(edge_attr.size()[0] // ((len(list(df_X.columns))) * 2)), :] * scale_factor + scale_min)
    actual_preds[:, :5] = actual_vals[:, :5]
    min_counts = np.min(actual_preds, axis=0)

    # Domain Shift Approach 
    # for i in range(5, 57):
    #     actual_preds[:, i] += abs(min_counts[i]) + 1e-8 if min_counts[i] <= 0 else 0

    for i in range(5, (len(list(df_X.columns)))):
        for j in range((edge_attr.size()[0] // ((len(list(df_X.columns))) * 2))):
            actual_preds[j, i] = args.min_proba if actual_preds[j, i] <= 0 else actual_preds[j, i]

    for i in range((edge_attr.size()[0] // ((len(list(df_X.columns))) * 2))):
        actual_preds[i, 5:] /= 1 if np.sum(actual_preds[i, 5:]) == 0 else np.sum(actual_preds[i, 5:])
    np.savetxt(FOLDER_PATH + 'data_pred.txt', actual_preds)

    data = []
    cause = []
    columns = []
    with open(glob.glob(f"uci/raw_data/{UCI_DATA}/data/{args.csv_data_file_name}")[0]) as fl:
        for line_index, line in enumerate(fl):
            if line_index == 0:
                columns = [item.replace("\n", "").replace(",", "") 
                           for item in line.split(",")]
                columns = columns[1:] if columns[0] == "" else columns
                columns = columns[:-1] if columns[-1] == "total" else columns
                continue
            data_row = [item.replace("\n", "").replace(",", "") 
                        for item in line.split(",")]
            if data_row != []:
                cause.append(data_row[2])
                prob_data_row = data_row[1:] if columns[0] == "" else data_row
                prob_data_row = data_row[:-1] if columns[-1] == "total" else data_row
                prob_data_row.append(0)
                data.append(prob_data_row)

    cause_dict = {item_index: item for item_index, item in enumerate(sorted(list(set(cause))))}
    cause = []
    data = []

    actual_preds_list = [list(actual_preds[i, :]) for i in range(actual_preds.shape[0])]
    print(len(actual_preds_list), len(actual_preds_list[0]))
    actual_preds_list = [[int(actual_preds_list[i][j])
                         if j < 5 else actual_preds_list[i][j]
                         for j in range(len(actual_preds_list[0]))]
                         for i in range(len(actual_preds_list))]
    
    actual_preds_list = [[cause_dict[actual_preds_list[i][j]]
                         if j == 1 else actual_preds_list[i][j]
                         for j in range(len(actual_preds_list[0]))]
                         for i in range(len(actual_preds_list))]
    
    if UCI_DATA == "en_matrix":
        ncode_path = f"uci/raw_data/{UCI_DATA}/data/ncode.csv"
        ncode_dict = {}
        with open(ncode_path, "r") as fl:
            for line_i, line in enumerate(fl):
                if line_i == 0:
                    continue
                line_clean = [item.replace("\n", "").replace(",", "").replace("\t", "").replace('"', "") for item in line.split("\t")]
                ncode_dict[line_clean[0]] = line_clean[1]

        columns = [f"{col}({ncode_dict[col]})" if col in ncode_dict else col for col in columns]
        
    df = pd.DataFrame(actual_preds_list, columns = columns)
    df.to_csv(FOLDER_PATH + 'data_pred.csv', index=False) 