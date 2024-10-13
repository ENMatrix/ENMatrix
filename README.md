# Self-Supervised Probability Imputation to Estimate the External-Natural Cause of Injury Matrix

## About

This repository is based on the "Handling Missing Data with Graph Representation Learning". [[**Project webpage**](http://snap.stanford.edu/grape/)].

This framework can be used to prepare the EN Matrix, in which the probabilities of natural causes of injuries are provided for external causes of injuries. This approach relies on [PyTorch](https://pytorch.org/) and [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html) that can be more efficiently used with GPUs.

## Results
Experiment results of self-supervised loss (L~D~) with various scale factors Ө~D~. Self-supervised loss (L~D~) was created based on the observation that among certain age groups, NI probabilities are very similar for ECI. This loss is computed using Euclidean of NI probabilities for similar age groups.
![Experiment results of self-supervised loss (L~D~) with various scale factors Ө~D~. Self-supervised loss (L~D~) was created based on the observation that among certain age groups, NI probabilities are very similar for ECI. This loss is computed using Euclidean of NI probabilities for similar age groups.](https://github.com/ENMatrix/ENMatrix/blob/main/Self_Supervised_Experiments.png)

Matrix imputation benchmarkin gresults.
![Matrix imputation benchmarking results.](https://github.com/ENMatrix/ENMatrix/blob/main/Bench_Marking.png)
## Installation

At the root folder:

```bash
conda env create -f environment.yml
conda activate grape
```

## Usage of EN Matrix

Step 1 - Prepare data (needs to be run only once for new data):

1- Our probablity dataset can be seen in repo addreess "uci/raw_data/en_matrix/data/en_appended_2022_proba.csv"
2- Navigate to the root folder of the repo
3- Type below in command line:

```bash
cd uci/raw_data/en_matrix/data/
python csv_to_txt_proba.py --csv_data_file_name {file_name}.csv 
python gen_test_valid_masks.py --csv_data_file_name {file_name}.csv 
```

This step generates the data.txt file that is used to train the model. It also generates a randomly created test set using file test_nan_mask.txt. Running steps above overwrites the data.txt and test_nan_mask.txt.

Step 2 - Pre-training:

1- Navigate to the root folder of the repo
2- Type below in command line:

```bash
python train_mdi.py --weight_decay 0.0000001 --lr 0.00125 --known 0.425 --valid 0.1 --save_model --epochs 50100 --opt_scheduler step --opt_decay_step 800 --opt_decay_rate 0.94 --log_dir {folder_name} --node_dim 256 --edge_dim 256 --impute_hiddens 256 --aggr mean --dropout 0.05 --dist_loss_cold_start_delay 50000 --dist_loss_delta 0.0002 --dist_loss_proba 1.00 --dist_loss_iter 10 --pre_training uci --data en_matrix --train_edge 0.9
```

The hyperparameters above can be tuned if necessary (helpful for new datasets):

\--weight_decay : Is a form of regularization on DNNs. This is used by the optimizer, [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html), which can reduce model overfitting. Please read the [article](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/) to learn about weight decay.

\--lr : This is the initial learning rate for training the Grapgh Neural Network (GNN). This parameter is also used by the optimizer [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html). Please read the [article](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/) to learn about leaning rates on GNNs.

\--known : Is an effective form of regularization on GNNs. This is one of the most important hyperparameters to tune, as it is very impactful on the model performance. This parameter is used in the beginning of every [epoch](https://deepai.org/machine-learning-glossary-and-terms/epoch) to randomly select the portion of cells to learn from (e.g. --known 0.425 randomly selects 42.5% of cells form the dataset to train the model). The systematic way to select hyperparameters is using grid search, but it requires many resources. Alternatively, you can rely on hand tuning to see find the optimum value for this hyperparameter. Hand tuning is a trial-and-error approach, in which you select a number, train the model, see performance, reduce it, see performance, repeat if performance improves, increase it if it doesn't, see performance, and repeat if performance improves. Hand tuning is an imperfect approach, as it assumes linear behavior which can be incorrect sometimes, but it is resource efficient and often works well in practice.

\--valid : Set it larger than zero to hold out a validation set. This is especially helpful in hyperparameters tuning, as it would be cheating to update hyperparameters based test set's performance. It automatically selects 0.125 of the dataset cells for valid set if it is set larger than 0.

\--save_model : If set, it would save the final model for future use. This should be set when it is used to fine-tune and predict the matrix after training.

\--epochs : Is the number of [epochs](https://deepai.org/machine-learning-glossary-and-terms/epoch) the model will be training. Tuning this parameter is often dependent on number of other parameters involving learning rate. Looking at the curves.png file in the "uci/test/en_matrix/{folder_name}" would help if the model has reached flattened learning curve. If curves.png is showing the model is still learning when the --epoch number is reached, you should increase this number, but if the model is stable in the end and it has stopped learning for tens of thousends of epochs, you should reduce the --epochs value.

\--opt_scheduler : This is the learning rate scheduler (e.g. [step](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)). Using the scheduler the leaning rate can be decayed, as GNN/DNN can stop learning on one leaning rate, but the same model can learn if the learning rate is decrease. However, decaying learning before the model has reached learning saturation at a given learning rate can reduce model performance.

\--opt_decay_step : Using [step](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR) the --opt_decay_step is assigned to the step_size parameter. This is the number of epochs before learning rate reduction is applied to the current learning rate.

\--opt_decay_rate : Using [step](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR) the --opt_decay_rate is assigned to the gamma parameter. This is the multiplicative factor of learning rate decay. Every time --opt_decay_step counter is reached the current learning is multiplied by --opt_decay_rate and set to the current learning rat value (e.g. using current hyperparameters after 1600 epochs current learning rate is 0.00125 x 0.94 x 0.94).

\--log_dir :  The results will be saved in "uci/test/en_matrix/{folder_name}". This should be set to a new folder name everytime.

\--node_dim : The embedding size of node (GNN model architecture parameter). Setting this parameter too large will drastically slow the training or maximize the available memory on the GPU. However, with proper regularization the model can have better performance with larger embedding sizes. This also is not always true. Sometimes smaller node embedding sizes outperform larger sizes, so it will require tuning.

\--edge_dim : The embedding size of edge (GNN model architecture parameter). Setting this parameter too large will drastically slow the training or maximize the available memory on the GPU. However, with proper regularization the model can have better performance with larger embedding sizes. This also is not always true. Sometimes smaller edge embedding sizes outperform larger sizes, so it will require tuning.

\--impute_hiddens : The hidden size of edge (GNN imputation model architecture parameter). Setting this parameter too large will drastically slow the training or maximize the available memory on the GPU. However, with proper regularization the model can have better performance with larger hidden sizes. This also is not always true. Sometimes smaller hidden sizes outperform larger sizes, so it will require tuning.

\--aggr : The aggregation function used for GNNs. Please read [article](https://medium.com/@pytorch_geometric/a-principled-approach-to-aggregations-983c086b10b3) for more details on aggregation function.

\--dropout : Is an effective form of regularization on GNNs. This is the embedding [dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html). Setting the parameter too large can make the model training very unstable, but an appropriate amount can improve training and model performance.

\--dist_loss_cold_start_delay : Is the number of epoch the model is trained for before the distance loss, which is a self-supervised loss function that forces the predictions of probabilities to be more simillar for rows that fall in similar age groups {[0, 1], [5, 10, 15], [20, 25, 30, 35], [40, 45, 50, 55, 60, 65, 70], [75, 80, 85, 90, 95+]}. This loss is used with the mean square error (MSE) loss. Before the --dist_loss_cold_start_delay number of epoch is reached, the loss function for training is MSE loss, but after the --dist_loss_cold_start_delay is reached, with some probability and some number of epochs the loss function becomes MSE + --dist_loss_delta * dist_loss. The distance loss is ~20 times slower in our test, so running it all the time is impractical, but our experiments have shown that after --dist_loss_cold_start_delay 50000 with --dist_loss_delta 0.002, the distance loss can reduce the distance by 20% in less than 100 epoch without noticeably degrading RMSE on valid set.

\--dist_loss_delta : Before the --dist_loss_cold_start_delay number of epoch is reached, the loss function for training is MSE loss, but after the --dist_loss_cold_start_delay is reached, with some probability and some number of epochs the loss function becomes MSE + --dist_loss_delta * dist_loss. If the --dist_loss_delta is set large it can overwhelm the MSE loss, so it is important to scale it. Past experiments have shown 0.002 can reduce the age group distances for similar ages by 20% in less than 100 epoch without noticeably degrading RMSE on valid set. If reducing the age group distances is more important than the RMSE, larger numbers than 0.002 can be selected, but it will impact the RSME on valid and test set.

\--dist_loss_proba and --dist_loss_iter : The distance loss can be applied intermittently as well. In this case, after --dist_loss_cold_start_delay number of epoch is reached, the distance loss is used in addition to MSE loss with --dist_loss_proba probablity and repeated for --dist_loss_iter iteration before the loss is switched back to MSE loss only. At every subsequent epoch distance loss can be used with --dist_loss_proba probablity and repeated for --dist_loss_iter iteration before the loss is switched back to MSE loss only. Our test showed that intermittent usage distance loss might not be as effective as using it continuously after the --dist_loss_cold_start_delay has reached, but it is available if dataset changes such that the intermittent distance loss becomes more effective.

\--pre_trainin : Generates 7 version of the original dataset, including (1) orginal data counterfactual data augmentation on (2) sex, (3) age (within similar age groups), (4) income (high income, not high income), and (5) platform (inpatient, outpatient), as well as weighted average data augmentation for (6) similar age groups and (7) similar age groups combined with sex. During pretraining there is an additional model that generates probabilities for each of the 7 datasets to be used in each epoch. Based on the generated probability a dataset is selected by sampling from the multinomial distribution at the beginning of every epoch. The probability generator model weights are updated using the same loss that updated the the GNN and imputation models. In our experiments the probabilities start unequal, but as training continues (10000 epochs), the model generates the same probability for all datasets to be selected, which is logical since if the model starts to overfit to any one dataset, its loss will increase on all other datasets, so it has to learn from all datasets equally. This approach of self-supervised pre-training helps improve model performance and it is more effective when labels are noisy, as the EN matrix suffers from sparsity, which makes the labels unreliable at times.

uci : GRAPE variable. Please do not change it.

\--data : The folder to use in "uci/raw_data/{--data}" for input data.

\--train_edge : Please set to a non-zero number, but this value is overwritten by the "test_nan_mask.txt" for test size, and the 0.125 valid size that has been hard coded for when --valid is larger than 0.

Step 3 - Fine-tuning:

This step is important to train the model on the finer subtleties of the original data.

1- Navigate to the root folder of the repo
2- Type below in command line:

```bash
python train_mdi.py --weight_decay 0.0000001 --lr 0.00035 --known 0.425 --valid 0.1 --save_model --epochs 20100 --opt_scheduler step --opt_decay_step 250 --opt_decay_rate 0.94 --log_dir {folder_name} --node_dim 256 --edge_dim 256 --impute_hiddens 256 --aggr mean --dropout 0.05 --dist_loss_cold_start_delay 20000 --dist_loss_delta 0.0002 --dist_loss_proba 1.00 --dist_loss_iter 10 --transfer_dir {transfer_folder_name} uci --data en_matrix --train_edge 0.9
```

The hyperparameters above can be tuned if necessary (helpful for new datasets):

\--lr : Has to be much smaller than starting learning rate of the pre-training to avid overwriting all that the model has learned in pre-training (e.g. in pre-training current learning rate of 0.00035 is reached after 16000 epochs 0.00125 * 0.94 \*\* (16000 / 800) ~ 0.00036).

\--transfer_dir : Has to be set to the folder name selected during pre-training.

Step 4 - Imputation:

This step creates the model generated matrix.

1- Navigate to the root folder of the repo
2- Type below in command line:

```bash
python imputation_check_proba_norm.py --uci_data en_matrix --model_folder_name {model folder name} --min_proba 1e-8 --csv_data_file_name en_appended_2022_proba.csv
```

\--uci_data : Use en_matrix.

\--model_folder_name : The {folder_name} used during fine-tuning the model.

\--min_proba : The probablity assigned to model predictions less than or equal to zero.

\--csv_data_file_name : Orginal data csv file used in step 1.