import numpy as np
import torch
import torch.nn.functional as F
import pickle
import copy, time
import random

from models.gnn_model import get_gnn, DatasetProbaAEModel
from models.prediction_model import MLPNet
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge


def mean_distnace_loss(data_pred, data_label, loc_mask, cluster_index_lsts, device, 
                       orig_matrix_num_rows, orig_matrix_num_cols, epoch, log_path):
    start_time = time.time()
    loc_mask = loc_mask.cpu()
    data_pred = data_pred.cpu()
    data_label = data_label.cpu()
    matrix_pred = torch.zeros(orig_matrix_num_rows, orig_matrix_num_cols).cpu()
    matrix_label = torch.zeros(orig_matrix_num_rows, orig_matrix_num_cols).cpu()
    for loc_index, loc in enumerate(loc_mask):
        matrix_pred[loc[0], loc[1]] = data_pred[loc_index]
        matrix_label[loc[0], loc[1]] = data_label[loc_index]
    matrix_pred = matrix_pred.to(device)
    matrix_label = matrix_label.to(device)
    cluster_mean = torch.zeros(len(cluster_index_lsts)).to(device)
    for cluster_index_lst_index, cluster_index_lst in enumerate(cluster_index_lsts):
        cluster_mean[cluster_index_lst_index] = torch.mean(torch.cdist(matrix_pred[torch.tensor(cluster_index_lst)][:, 5:], 
                                                                       matrix_pred[torch.tensor(cluster_index_lst)][:, 5:]))
    ret = torch.mean(cluster_mean)
    print("mean_distnace_loss took: ", time.time() - start_time, " Seconds")
    return ret


def dataset_prep(args, data, index_tracking):
    device = "cpu"
    x = data.x.clone().detach().to(device)
    if hasattr(args,'split_sample') and args.split_sample > 0.:
        if args.split_train:
            all_train_edge_index = data.lower_train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.lower_train_edge_attr.clone().detach().to(device)
            all_train_labels = data.lower_train_labels.clone().detach().to(device)
        else:
            all_train_edge_index = data.train_edge_index.clone().detach().to(device)
            all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
            all_train_labels = data.train_labels.clone().detach().to(device)
        if args.split_test:
            test_input_edge_index = data.higher_train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.higher_train_edge_attr.clone().detach().to(device)
        else:
            test_input_edge_index = data.train_edge_index.clone().detach().to(device)
            test_input_edge_attr = data.train_edge_attr.clone().detach().to(device)
        test_edge_index = data.higher_test_edge_index.clone().detach().to(device)
        test_edge_attr = data.higher_test_edge_attr.clone().detach().to(device)
        test_labels = data.higher_test_labels.clone().detach().to(device)
    else:
        all_train_edge_index = data.train_edge_index.clone().detach().to(device)
        all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
        all_train_labels = data.train_labels.clone().detach().to(device)
        test_input_edge_index = all_train_edge_index
        test_input_edge_attr = all_train_edge_attr
        test_edge_index = data.test_edge_index.clone().detach().to(device)
        test_edge_attr = data.test_edge_attr.clone().detach().to(device)
        test_labels = data.test_labels.clone().detach().to(device)
    if hasattr(data,'class_values'):
        class_values = data.class_values.clone().detach().to(device)
    else:
        class_values = torch.tensor([])
    first_five_mask_train = torch.tensor(copy.deepcopy(np.array([True if item[1] < 5 else False for item in index_tracking])), 
                                         dtype=torch.bool, requires_grad=False).to(device)
    first_five_mask_valid = torch.tensor(copy.deepcopy(np.array([True if item[1] < 5 else False for item in index_tracking])), 
                                         dtype=torch.bool, requires_grad=False).to(device)
    index_tracking_test_valid_removed = torch.tensor(copy.deepcopy(index_tracking), 
                                                     dtype=torch.int16, requires_grad=False).to(device)
    index_tracking = torch.tensor(index_tracking, 
                                  requires_grad=False, dtype=torch.int16).to(device)
    if args.valid > 0.:
        valid_mask = get_known_mask(0.143, int(all_train_edge_attr.shape[0] / 2)).to(device)
        # print(np.loadtxt("../../comp/valid_nan_mask.txt").shape, np.loadtxt("../../comp/valid_nan_mask.txt").flatten().shape)
        # valid_mask = torch.tensor(np.isnan(np.loadtxt("../../comp/valid_nan_mask.txt").flatten())).to(device)
        # np.savetxt("valid_mask.txt", valid_mask.cpu().numpy())
        print("valid mask sum: ",torch.sum(valid_mask))
        train_labels = all_train_labels[~valid_mask]
        print("all_train_labels", all_train_labels.shape)
        print("train_labels", train_labels.shape)
        first_five_mask_train = first_five_mask_train[~valid_mask]
        index_tracking_test_valid_removed = index_tracking[~valid_mask]
        print("first_five_mask_train", first_five_mask_train.size())
        print("index_tracking after valid filtering", index_tracking.shape)
        valid_labels = all_train_labels[valid_mask]
        first_five_mask_valid = first_five_mask_valid[valid_mask]
        print("first_five_mask_valid", first_five_mask_valid.size())
        double_valid_mask = torch.cat((valid_mask, valid_mask), dim=0)
        valid_edge_index, valid_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, double_valid_mask, True)
        train_edge_index, train_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, ~double_valid_mask, True)
        print("train edge num is {}, valid edge num is {}, test edge num is input {} output {}"\
                .format(
                train_edge_attr.shape[0], valid_edge_attr.shape[0],
                test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
    else:
        train_edge_index, train_edge_attr, train_labels =\
             all_train_edge_index, all_train_edge_attr, all_train_labels
        print("train edge num is {}, test edge num is input {}, output {}"\
                .format(
                train_edge_attr.shape[0],
                test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
    if args.auto_known:
        args.known = float(all_train_labels.shape[0])/float(all_train_labels.shape[0]+test_labels.shape[0])
        print("auto calculating known is {}/{} = {:.3g}".format(all_train_labels.shape[0],all_train_labels.shape[0]+test_labels.shape[0],args.known))
    return train_edge_attr, train_edge_index, train_labels, first_five_mask_train, x, index_tracking_test_valid_removed,\
           valid_edge_index, class_values, valid_labels, valid_edge_attr, first_five_mask_valid,\
           test_edge_index, test_edge_attr, test_labels, test_input_edge_attr, test_input_edge_index
    
def prep_all_datasets(args, datasets, index_tracking):
    train_edge_attr_lst, train_edge_index_lst, train_labels_lst, first_five_mask_train_lst, x_lst, index_tracking_test_valid_removed_lst,\
          valid_edge_index_lst, class_values_lst, valid_labels_lst, valid_edge_attr_lst, first_five_mask_valid_lst,\
          test_edge_index_lst, test_edge_attr_lst, test_labels_lst, test_input_edge_attr_lst, test_input_edge_index_lst\
              = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    print(f"There are {len(datasets)} augmented datasets to train this model.")
    for dataset_index, dataset in enumerate(datasets):
        print(f"Preparing Dataset {dataset_index}")
        train_edge_attr, train_edge_index, train_labels, first_five_mask_train, x, index_tracking_test_valid_removed,\
          valid_edge_index, class_values, valid_labels, valid_edge_attr, first_five_mask_valid,\
          test_edge_index, test_edge_attr, test_labels, test_input_edge_attr, test_input_edge_index = dataset_prep(args, dataset, index_tracking)
        train_edge_attr_lst.append(train_edge_attr)
        train_edge_index_lst.append(train_edge_index)
        train_labels_lst.append(train_labels)
        first_five_mask_train_lst.append(first_five_mask_train)
        x_lst.append(x)
        index_tracking_test_valid_removed_lst.append(index_tracking_test_valid_removed)
        valid_edge_index_lst.append(valid_edge_index)
        class_values_lst.append(class_values)
        valid_labels_lst.append(valid_labels)
        valid_edge_attr_lst.append(valid_edge_attr)
        first_five_mask_valid_lst.append(first_five_mask_valid)
        test_edge_index_lst.append(test_edge_index)
        test_edge_attr_lst.append(test_edge_attr)
        test_labels_lst.append(test_labels)
        test_input_edge_attr_lst.append(test_input_edge_attr)
        test_input_edge_index_lst.append(test_input_edge_index)
    return (train_edge_attr_lst, train_edge_index_lst, train_labels_lst, first_five_mask_train_lst, x_lst, index_tracking_test_valid_removed_lst,\
            valid_edge_index_lst, class_values_lst, valid_labels_lst, valid_edge_attr_lst, first_five_mask_valid_lst,\
            test_edge_index_lst, test_edge_attr_lst, test_labels_lst, test_input_edge_attr_lst, test_input_edge_index_lst)
    
def write_dataset_to_gpu(index, dataset_output_lists, device):
    train_edge_attr_lst, train_edge_index_lst, train_labels_lst, first_five_mask_train_lst, x_lst, index_tracking_test_valid_removed_lst,\
        valid_edge_index_lst, class_values_lst, valid_labels_lst, valid_edge_attr_lst, first_five_mask_valid_lst,\
        test_edge_index_lst, test_edge_attr_lst, test_labels_lst, test_input_edge_attr_lst, test_input_edge_index_lst = dataset_output_lists
    return train_edge_attr_lst[index].detach().clone().to(device),\
            train_edge_index_lst[index].detach().clone().to(device),\
            train_labels_lst[index].detach().clone().to(device),\
            first_five_mask_train_lst[index].detach().clone().to(device),\
            x_lst[index].detach().clone().to(device),\
            index_tracking_test_valid_removed_lst[index].detach().clone().to(device),\
            valid_edge_index_lst[index].detach().clone().to(device),\
            class_values_lst[index].detach().clone().to(device),\
            valid_labels_lst[index].detach().clone().to(device),\
            valid_edge_attr_lst[index].detach().clone().to(device),\
            first_five_mask_valid_lst[index].detach().clone().to(device),\
            test_edge_index_lst[index].detach().clone().to(device),\
            test_edge_attr_lst[index].detach().clone().to(device),\
            test_labels_lst[index].detach().clone().to(device),\
            test_input_edge_attr_lst[index].detach().clone().to(device),\
            test_input_edge_index_lst[index].detach().clone().to(device)

def remove_dataset_from_gpu(dataset_outputs):
    train_edge_attr, train_edge_index, train_labels, first_five_mask_train, x, index_tracking_test_valid_removed,\
          valid_edge_index, class_values, valid_labels, valid_edge_attr, first_five_mask_valid,\
          test_edge_index, test_edge_attr, test_labels, test_input_edge_attr, test_input_edge_index = dataset_outputs
    del train_edge_attr
    del train_edge_index
    del train_labels
    del first_five_mask_train
    del x
    del index_tracking_test_valid_removed
    del valid_edge_index
    del class_values
    del valid_labels
    del valid_edge_attr
    del first_five_mask_valid
    del test_edge_index
    del test_edge_attr
    del test_labels
    del test_input_edge_attr
    del test_input_edge_index
    torch.cuda.empty_cache()


def train_gnn_mdi(datasets, args, log_path, index_tracking,  cluster_index_lsts, device=torch.device('cpu')):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dist_loss_delta = torch.tensor(args.dist_loss_delta).to(device)
    dist_loss_proba = args.dist_loss_proba
    dist_loss_iter = args.dist_loss_iter
    dist_loss_cold_start_delay = args.dist_loss_cold_start_delay
    # Preparing datasets
    print(f"There are {len(datasets)} augmented datasets to train this model.")
    dataset_output_lists = prep_all_datasets(args, datasets, index_tracking)
    data = datasets[0]
    proba_dataset_model = DatasetProbaAEModel(len(datasets), len(datasets)).to(device)
    model = get_gnn(data, args).to(device)
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2
    if hasattr(args,'ce_loss') and args.ce_loss:
        output_dim = len(data.class_values)
    else:
        output_dim = 1
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            dropout=args.dropout).to(device)
    if args.transfer_dir: # this ensures the valid mask is consistant
        load_path = './{}/test/{}/{}/'.format(args.domain,args.data,args.transfer_dir)
        print("loading fron {} with {}".format(load_path,args.transfer_extra))
        model = torch.load(load_path+'model'+args.transfer_extra+'.pt', map_location=torch.device('cpu'))
        model = model.to(device)
        impute_model = torch.load(load_path+'impute_model'+args.transfer_extra+'.pt', map_location=torch.device('cpu'))
        impute_model = impute_model.to(device)
    for param in proba_dataset_model.parameters():
        param.requires_grad = True
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)\
                          + sum(p.numel() for p in impute_model.parameters() if p.requires_grad)\
                          + sum(p.numel() for p in proba_dataset_model.parameters() if p.requires_grad)
    print("total trainable_parameters: ",trainable_parameters)
    # build optimizer
    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters())\
                           + list(proba_dataset_model.parameters())
    scheduler, opt = build_optimizer(args, trainable_parameters)
    # train
    Train_loss = []
    Test_rmse = []
    Test_l1 = []
    Lr = []
    if args.valid > 0.:
        Valid_rmse = []
        Valid_l1 = []
        best_valid_rmse = np.inf
        best_valid_rmse_epoch = 0
        best_valid_l1 = np.inf
        best_valid_l1_epoch = 0
    obj = dict()
    obj['args'] = args
    obj['outputs'] = dict()

    probas_init = (torch.tensor([1.0] * len(datasets)) / torch.sum(torch.tensor([1.0] * len(datasets)))).to(device)
    probas = list(probas_init.clone().detach().cpu().numpy())
    dist_loss_count = dist_loss_iter
    orig_matrix_num_rows = max([item[0] for item in index_tracking]) + 1
    orig_matrix_num_cols = max([item[1] for item in index_tracking]) + 1
    # Training Loop 
    for epoch in range(args.epochs):
        model.train()
        impute_model.train()
        proba_dataset_model.train()

        print(f"Dataset selection probabilities are: {probas}")
        dataset_selection = random.choices(population=list(range(len(datasets))), 
                                           weights=probas, 
                                           k=1)[0]
        print(f"dataset {dataset_selection} is selected for this epoch")
        train_edge_attr, train_edge_index, train_labels, first_five_mask_train, x, index_tracking_test_valid_removed,\
          valid_edge_index, class_values, valid_labels, valid_edge_attr, first_five_mask_valid,\
          test_edge_index, test_edge_attr, test_labels, test_input_edge_attr, test_input_edge_index = \
            write_dataset_to_gpu(dataset_selection, dataset_output_lists, device)
        
        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        if hasattr(args,'ce_loss') and args.ce_loss:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
            print("ERROR: hasattr(args,'ce_loss') and args.ce_loss is true")
        else:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
        if args.loss_mode == 1:
            pred_train[known_mask] = train_labels[known_mask]
        label_train = train_labels
        # first five colls copy labels to remove impact from train loss
        # print(known_mask.size())
        pred_train[first_five_mask_train] = train_labels[first_five_mask_train]
        if hasattr(args,'ce_loss') and args.ce_loss:
            loss = F.cross_entropy(pred_train,train_labels)

        else:
            sample = random.uniform(0, 1)
            if  epoch >= dist_loss_cold_start_delay and (sample <= dist_loss_proba or dist_loss_count < dist_loss_iter):
                dist_loss = mean_distnace_loss(pred_train, label_train, index_tracking_test_valid_removed, cluster_index_lsts, device, 
                                               orig_matrix_num_rows, orig_matrix_num_cols, epoch, log_path)
                
                x_embd = model(x, known_edge_attr, known_edge_index)
                pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
                pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
                pred_train[first_five_mask_train] = train_labels[first_five_mask_train]
                train_mse_loss = F.mse_loss(pred_train, label_train)
                loss = train_mse_loss + (dist_loss_delta * dist_loss)
                if sample <= dist_loss_proba:
                    dist_loss_count = 0
                dist_loss_count += 1
            else:
                dist_loss = 0.0
                train_mse_loss = F.mse_loss(pred_train, label_train)
                loss = train_mse_loss

        if args.pre_training:
            dataset_selection_proba = torch.nn.Softmax(dim=0)(proba_dataset_model(probas_init))
            probas = list(dataset_selection_proba.clone().detach().cpu().numpy())
            loss = (dataset_selection_proba * loss).norm(p=1)

        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        impute_model.eval()
        proba_dataset_model.eval()
        with torch.no_grad():
            if args.valid > 0.:
                x_embd = model(x, train_edge_attr, train_edge_index)
                pred = impute_model([x_embd[valid_edge_index[0], :], x_embd[valid_edge_index[1], :]])
                if hasattr(args,'ce_loss') and args.ce_loss:
                    pred_valid = class_values[pred[:int(valid_edge_attr.shape[0] / 2)].max(1)[1]]
                    label_valid = class_values[valid_labels]
                elif hasattr(args,'norm_label') and args.norm_label:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    pred_valid = pred_valid * max(class_values)
                    label_valid = valid_labels
                    label_valid = label_valid * max(class_values)
                else:
                    pred_valid = pred[:int(valid_edge_attr.shape[0] / 2),0]
                    label_valid = valid_labels
                    pred_valid[first_five_mask_valid] = valid_labels[first_five_mask_valid]
                mse = F.mse_loss(pred_valid, label_valid)
                valid_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_valid, label_valid)
                valid_l1 = l1.item()
                if valid_l1 < best_valid_l1:
                    best_valid_l1 = valid_l1
                    best_valid_l1_epoch = epoch
                    if args.save_model:
                        torch.save(model, log_path + 'model_best_valid_l1.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_l1.pt')
                if valid_rmse < best_valid_rmse:
                    best_valid_rmse = valid_rmse
                    best_valid_rmse_epoch = epoch
                    if args.save_model:
                        torch.save(model, log_path + 'model_best_valid_rmse.pt')
                        torch.save(impute_model, log_path + 'impute_model_best_valid_rmse.pt')
                Valid_rmse.append(valid_rmse)
                Valid_l1.append(valid_l1)

            x_embd = model(x, test_input_edge_attr, test_input_edge_index)
            pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
            if hasattr(args,'ce_loss') and args.ce_loss:
                pred_test = class_values[pred[:int(test_edge_attr.shape[0] / 2)].max(1)[1]]
                label_test = class_values[test_labels]
            elif hasattr(args,'norm_label') and args.norm_label:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                pred_test = pred_test * max(class_values)
                label_test = test_labels
                label_test = label_test * max(class_values)
            else:
                pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
                label_test = test_labels
            mse = F.mse_loss(pred_test, label_test)
            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(pred_test, label_test)
            test_l1 = l1.item()
            if args.save_prediction:
                if epoch == best_valid_rmse_epoch:
                    obj['outputs']['best_valid_rmse_pred_test'] = pred_test.detach().cpu().numpy()
                if epoch == best_valid_l1_epoch:
                    obj['outputs']['best_valid_l1_pred_test'] = pred_test.detach().cpu().numpy()

            if args.mode == 'debug':
                torch.save(model, log_path + 'model_{}.pt'.format(epoch))
                torch.save(impute_model, log_path + 'impute_model_{}.pt'.format(epoch))
            Train_loss.append(train_loss)
            Test_rmse.append(test_rmse)
            Test_l1.append(test_l1)
            print('epoch: ', epoch)
            print('MSE loss: ', train_mse_loss.item())
            print('Distance loss: ', dist_loss.item() if torch.is_tensor(dist_loss) else dist_loss)
            print('loss: ', train_loss)
            if args.valid > 0.:
                print('valid rmse: ', valid_rmse)
                print('valid l1: ', valid_l1)
            print('test rmse: ', test_rmse)
            print('test l1: ', test_l1)
        dataset_outputs = (train_edge_attr, train_edge_index, train_labels, first_five_mask_train, x, index_tracking_test_valid_removed,\
          valid_edge_index, class_values, valid_labels, valid_edge_attr, first_five_mask_valid,\
          test_edge_index, test_edge_attr, test_labels, test_input_edge_attr, test_input_edge_index)
        remove_dataset_from_gpu(dataset_outputs)
        

    pred_train = pred_train.detach().cpu().numpy()
    label_train = label_train.detach().cpu().numpy()
    pred_test = pred_test.detach().cpu().numpy()
    label_test = label_test.detach().cpu().numpy()

    obj['curves'] = dict()
    obj['curves']['train_loss'] = Train_loss
    if args.valid > 0.:
        obj['curves']['valid_rmse'] = Valid_rmse
        obj['curves']['valid_l1'] = Valid_l1
    obj['curves']['test_rmse'] = Test_rmse
    obj['curves']['test_l1'] = Test_l1
    obj['lr'] = Lr

    obj['outputs']['final_pred_train'] = pred_train
    obj['outputs']['label_train'] = label_train
    obj['outputs']['final_pred_test'] = pred_test
    obj['outputs']['label_test'] = label_test
    pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    if args.save_model:
        torch.save(model, log_path + 'model.pt')
        torch.save(impute_model, log_path + 'impute_model.pt')

    # obj = objectview(obj)
    plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
                clip=True, label_min=True, label_end=True)
    plot_curve(obj, log_path+'lr.png',keys=['lr'], 
                clip=False, label_min=False, label_end=False)
    plot_sample(obj['outputs'], log_path+'outputs.png', 
                groups=[['final_pred_train','label_train'],
                        ['final_pred_test','label_test']
                        ], 
                num_points=20)
    if args.save_prediction and args.valid > 0.:
        plot_sample(obj['outputs'], log_path+'outputs_best_valid.png', 
                    groups=[['best_valid_rmse_pred_test','label_test'],
                            ['best_valid_l1_pred_test','label_test']
                            ], 
                    num_points=20)
    if args.valid > 0.:
        print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse,best_valid_rmse_epoch))
        print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))
