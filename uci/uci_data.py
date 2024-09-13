import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing

import torch
import random, copy
import numpy as np
import pdb

from utils.utils import get_known_mask, mask_edge

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

def gen_cluster_index_lists(df_X):
    age_str = {1: "[0 to 1]",
            15: "(1 to 15]",
            35: "(15 to 35]",
            70: "(35 to 70]",
            95: "(70 to 95+]"}
    age_aggr_lst = [1, 15, 35, 70, 95]
    data_age_agrg = {}
    data_age_sex_agrg = {}
    data = df_X.values
    for data_row_index, data_row in enumerate(data):
        age_agr_row_id_tpl = ()
        age_sex_agr_row_id_tpl = ()
        for age_group_index, age_group in enumerate(age_aggr_lst):
            if data_row[0] <= age_group:
                age_agr_row_id_tpl = (age_aggr_lst[age_group_index], data_row[1], data_row[2], data_row[3], data_row[4])
                age_sex_agr_row_id_tpl = (age_aggr_lst[age_group_index], data_row[1], data_row[2], data_row[4])
                break
        if age_agr_row_id_tpl in data_age_agrg:
            data_age_agrg[age_agr_row_id_tpl].append(data_row_index)
            
        elif age_agr_row_id_tpl not in data_age_agrg:
            data_age_agrg[age_agr_row_id_tpl] = [data_row_index]
        
        if age_sex_agr_row_id_tpl in data_age_sex_agrg:
            data_age_sex_agrg[age_sex_agr_row_id_tpl].append(data_row_index)

        elif age_sex_agr_row_id_tpl not in data_age_sex_agrg:
            data_age_sex_agrg[age_sex_agr_row_id_tpl] = [data_row_index]
    print(list(data_age_agrg.values()))
    return list(data_age_agrg.values()), list(data_age_sex_agrg.values())

def get_data(df_X, df_y, node_mode, train_edge_prob, split_sample_ratio, split_by, train_y_prob, uci_path, args, seed=0, normalize=True):
    if len(df_y.shape)==1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape)==2:
        df_y = df_y[0].to_numpy()
    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
        print(df_X.head())
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = create_node(df_X, node_mode) 
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)
    # set seed to fix known/unknwon edges
    torch.manual_seed(seed)
    # keep train_edge_prob of all edges
    train_edge_mask_matrix = torch.tensor(~np.isnan(np.loadtxt(uci_path+'/raw_data/{}/data/test_nan_mask.txt'.format(args.data))))
    train_edge_mask = torch.tensor(~np.isnan(np.loadtxt(uci_path+'/raw_data/{}/data/test_nan_mask.txt'.format(args.data))).flatten())


    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)

    # mask edges based on the generated train_edge_mask
    # train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                  double_train_edge_mask, True)
    
    print("train_edge_index", train_edge_index.shape)
    print("train_edge_attr", train_edge_attr.shape)
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2),0]
    print("train_labels", train_labels.shape)
    index_tracking = np.array([(i, j) 
                               for i in range(train_edge_mask_matrix.size()[0]) 
                               for j in range(train_edge_mask_matrix.size()[1])])
    
    print("index_tracking prefilter", len(index_tracking))
    index_tracking = index_tracking[train_edge_mask]
    print("index_tracking filtered", len(index_tracking))


    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    
    print("test_edge_index", test_edge_index.shape)
    print("test_edge_attr", test_edge_attr.shape)
    
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2),0]
    print("test_labels", test_labels.shape)

    # mask the y-values during training, i.e. how we split the training and test sets
    train_y_mask = np.ones(y.shape[0]) == 1 # get_known_mask(train_y_prob, y.shape[0])

    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
            train_y_mask=train_y_mask, test_y_mask=test_y_mask,
            train_edge_index=train_edge_index,train_edge_attr=train_edge_attr,
            train_edge_mask=train_edge_mask,train_labels=train_labels,
            test_edge_index=test_edge_index,test_edge_attr=test_edge_attr,
            test_edge_mask=~train_edge_mask,test_labels=test_labels, 
            df_X=df_X,df_y=df_y,
            edge_attr_dim=train_edge_attr.shape[-1],
            user_num=df_X.shape[0]
            )

    if split_sample_ratio > 0.:
        if split_by == 'y':
            sorted_y, sorted_y_index = torch.sort(torch.reshape(y,(-1,)))
        elif split_by == 'random':
            sorted_y_index = torch.randperm(y.shape[0])
        lower_y_index = sorted_y_index[:int(np.floor(y.shape[0]*split_sample_ratio))]
        higher_y_index = sorted_y_index[int(np.floor(y.shape[0]*split_sample_ratio)):]
        # here we don't split x, only split edge
        # train
        half_train_edge_index = train_edge_index[:,:int(train_edge_index.shape[1]/2)];
        lower_train_edge_mask = []
        for node_index in half_train_edge_index[0]:
            if node_index in lower_y_index:
                lower_train_edge_mask.append(True)
            else:
                lower_train_edge_mask.append(False)
        lower_train_edge_mask = torch.tensor(lower_train_edge_mask)
        double_lower_train_edge_mask = torch.cat((lower_train_edge_mask, lower_train_edge_mask), dim=0)
        lower_train_edge_index, lower_train_edge_attr = mask_edge(train_edge_index, train_edge_attr,
                                                double_lower_train_edge_mask, True)
        lower_train_labels = lower_train_edge_attr[:int(lower_train_edge_attr.shape[0]/2),0]
        higher_train_edge_index, higher_train_edge_attr = mask_edge(train_edge_index, train_edge_attr,
                                                ~double_lower_train_edge_mask, True)
        higher_train_labels = higher_train_edge_attr[:int(higher_train_edge_attr.shape[0]/2),0]
        # test
        half_test_edge_index = test_edge_index[:,:int(test_edge_index.shape[1]/2)];
        lower_test_edge_mask = []
        for node_index in half_test_edge_index[0]:
            if node_index in lower_y_index:
                lower_test_edge_mask.append(True)
            else:
                lower_test_edge_mask.append(False)
        lower_test_edge_mask = torch.tensor(lower_test_edge_mask)
        double_lower_test_edge_mask = torch.cat((lower_test_edge_mask, lower_test_edge_mask), dim=0)
        lower_test_edge_index, lower_test_edge_attr = mask_edge(test_edge_index, test_edge_attr,
                                                double_lower_test_edge_mask, True)
        lower_test_labels = lower_test_edge_attr[:int(lower_test_edge_attr.shape[0]/2),0]
        higher_test_edge_index, higher_test_edge_attr = mask_edge(test_edge_index, test_edge_attr,
                                                ~double_lower_test_edge_mask, True)
        higher_test_labels = higher_test_edge_attr[:int(higher_test_edge_attr.shape[0]/2),0]


        data.lower_y_index = lower_y_index
        data.higher_y_index = higher_y_index
        data.lower_train_edge_index = lower_train_edge_index
        data.lower_train_edge_attr = lower_train_edge_attr
        data.lower_train_labels = lower_train_labels
        data.higher_train_edge_index = higher_train_edge_index
        data.higher_train_edge_attr = higher_train_edge_attr
        data.higher_train_labels = higher_train_labels
        data.lower_test_edge_index = lower_test_edge_index
        data.lower_test_edge_attr = lower_test_edge_attr
        data.lower_test_labels = lower_train_labels
        data.higher_test_edge_index = higher_test_edge_index
        data.higher_test_edge_attr = higher_test_edge_attr
        data.higher_test_labels = higher_test_labels
        
    return data, index_tracking


def get_data_gender_counter_factual(df_X):
    data = df_X.values
    cols = df_X.columns
    for row in data:
        if row[3] == 1.0:
            row[3] = 2.0
        elif row[3] == 2.0:
            row[3] = 1.0
        else:
            print("ERROR sex is not assigned as 1 or 2")
    return pd.DataFrame(data=data, columns=cols)

def get_data_income_counter_factual(df_X):
    data = df_X.values
    cols = df_X.columns
    for row in data:
        if row[4] == 0.0:
            row[4] = 1.0
        elif row[4] == 1.0:
            row[4] = 0.0
        else:
            print("ERROR high income is not assigned as 0 or 1")
    return pd.DataFrame(data=data, columns=cols)

def get_data_inpatient_counter_factual(df_X):
    data = df_X.values
    cols = df_X.columns
    for row in data:
        if row[2] == 0.0:
            row[2] = 1.0
        elif row[2] == 1.0:
            row[2] = 0.0
        else:
            print("ERROR inpatient is not assigned as 0 or 1")
    return pd.DataFrame(data=data, columns=cols)

def get_data_age_counter_factual(df_X):
    data = df_X.values
    cols = df_X.columns
    age_aggr_lst = [1, 15, 35, 70, 95]
    age_option_counter_factual = { 1: [0, 1],
                                  15: [5, 10, 15],
                                  35: [20, 25, 30, 35],
                                  70: [40, 45, 50, 55, 60, 65, 70],
                                  95: [75, 80, 85, 90, 95]}
    for row in data:
        for age_group in age_aggr_lst:
            if row[0] <= age_group:
                row[0] = random.choice(age_option_counter_factual[age_group])
                break
    return pd.DataFrame(data=data, columns=cols)

def get_data_aggreg_group_weighted_ave(df_X, cluster_index_lsts):
    data = df_X.values
    cols = df_X.columns
    for cluster_lst in cluster_index_lsts:
        zeros_per_row_lst = []
        for i in range(data[cluster_lst][:, 5:].shape[0]):
            zeros_per_row_lst.append(np.where(data[cluster_lst][:, 5:] == 0.0)[0].shape[0])
            data[cluster_lst[i], 5:] /= zeros_per_row_lst[-1] if zeros_per_row_lst[-1] > 0 else 1
        data[cluster_lst][:, 5:] = np.array([np.sum(data[cluster_lst][:, 5:], axis=0) / 
                                             (np.sum(data[cluster_lst][:, 5:]) 
                                             if np.sum(data[cluster_lst][:, 5:]) > 0.0 
                                             else 1)] * data[cluster_lst][:, 5:].shape[0])
    return pd.DataFrame(data=data, columns=cols)

def load_data(args):
    uci_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    df_np = np.loadtxt(uci_path+'/raw_data/{}/data/data.txt'.format(args.data))
    df_y = pd.DataFrame(df_np[:, -1:])
    print(df_y.head())
    df_X = pd.DataFrame(df_np[:, :-1])
    print(df_X.head())
    if not hasattr(args,'split_sample'):
        args.split_sample = 0

    cluster_age_index_lsts, cluster_age_sex_index_lsts = gen_cluster_index_lists(df_X)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # All 7 sets
    if args.pre_training:
        data_orig, index_tracking = get_data(copy.deepcopy(df_X), df_y, args.node_mode, 
                                             args.train_edge, args.split_sample, args.split_by, args.train_y, uci_path, args,
                                             args.seed, normalize=(args.not_normalized==0))
        data_gender_cf, index_tracking = get_data(get_data_gender_counter_factual(copy.deepcopy(df_X)), df_y, args.node_mode,
                                                  args.train_edge, args.split_sample, args.split_by, args.train_y, uci_path, args,
                                                  args.seed, normalize=(args.not_normalized==0))
        data_income_cf, index_tracking = get_data(get_data_income_counter_factual(copy.deepcopy(df_X)), df_y, args.node_mode,
                                                  args.train_edge, args.split_sample, args.split_by, args.train_y, uci_path, args,
                                                  args.seed, normalize=(args.not_normalized==0))
        data_inpatient_cf, index_tracking = get_data(get_data_inpatient_counter_factual(copy.deepcopy(df_X)), df_y, args.node_mode,
                                                     args.train_edge, args.split_sample, args.split_by, args.train_y, uci_path, args,
                                                     args.seed, normalize=(args.not_normalized==0))
        data_age_cf, index_tracking = get_data(get_data_age_counter_factual(copy.deepcopy(df_X)), df_y, args.node_mode,
                                               args.train_edge, args.split_sample, args.split_by, args.train_y, uci_path, args, 
                                               args.seed, normalize=(args.not_normalized==0))
        data_wa_age_aggr, index_tracking = get_data(get_data_aggreg_group_weighted_ave(copy.deepcopy(df_X), cluster_age_index_lsts), df_y, args.node_mode, 
                                                    args.train_edge, args.split_sample, args.split_by, args.train_y,  uci_path, args, 
                                                    args.seed, normalize=(args.not_normalized==0))
        data_wa_age_sex_aggr, index_tracking = get_data(get_data_aggreg_group_weighted_ave(copy.deepcopy(df_X), cluster_age_sex_index_lsts), df_y, args.node_mode,
                                                        args.train_edge, args.split_sample, args.split_by, args.train_y,  uci_path, args, 
                                                        args.seed, normalize=(args.not_normalized==0))
        datasets = [data_orig, data_gender_cf, data_income_cf, data_inpatient_cf, data_age_cf, data_wa_age_aggr, data_wa_age_sex_aggr]
    
    # Just the original dataset for fine-tuning 
    else:
        data_orig, index_tracking = get_data(copy.deepcopy(df_X), df_y, args.node_mode, args.train_edge, args.split_sample, args.split_by, args.train_y,
                                              uci_path, args, args.seed, normalize=(args.not_normalized==0))
        datasets = [data_orig]
    print(f"There are {len(datasets)} augmented datasets to train this model.")
    return datasets, index_tracking, cluster_age_index_lsts


