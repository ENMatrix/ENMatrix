import random
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_data_file_name', type=str, default='Provide CSV data file name path') 
args = parser.parse_args()
print(args)

data = []
e_cause = []

data_file_path = args.csv_data_file_name
with open(data_file_path) as fl:
    for line_index, line in enumerate(fl):
        if line_index == 0:
            continue
        data_row = [item.replace("\n", "").replace(",", "") 
                    for item in line.split(",") 
                    if int(line.split(",")[-1].replace("\n", "").replace(",", "")) >= 0]
        if data_row != []:
            e_cause.append(data_row[2])
            prob_data_row = data_row[1:]
            prob_data_row.append(0)
            data.append(prob_data_row)

e_cause_dict = {item: item_index for item_index, item in enumerate(sorted(list(set(e_cause))))}
e_cause = []

for row_index, row in enumerate(data):
    row[1] = e_cause_dict[row[1]]
    row = [float(item) for item in row]
    data[row_index] = row

data = np.array(data, dtype=np.float64)
print(data.shape)
rows_times_cols = [i for i in range(data.shape[0]) 
                   for _ in range(5, data.shape[1])]
cols_times_rows = [j for _ in range(data.shape[0]) 
                   for j in range(5, data.shape[1])]
print(len(rows_times_cols), len(cols_times_rows))
assert(len(rows_times_cols) == len(cols_times_rows))
index_tuples_lst = list(zip(rows_times_cols, cols_times_rows))
print(len(index_tuples_lst), len(set(index_tuples_lst)))
random.shuffle(index_tuples_lst)

valid_index_tuples = index_tuples_lst[:int(np.round(len(index_tuples_lst) * 0.125))]
test_index_tuples = index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.125)): int(np.round(len(index_tuples_lst) * 0.25))]
print(f"There are {len(valid_index_tuples)} validation values selected")
print(f"There are {len(test_index_tuples)} test values selected")

valid_mask = np.zeros((data.shape[0], data.shape[1]))
valid_rows, valid_cols = zip(*valid_index_tuples)
print(len(valid_rows), len(valid_cols))
valid_mask[valid_rows, valid_cols] = np.nan

print(f"Varifying: There are {int(np.sum(np.nan_to_num(valid_mask, nan=1.0)))} validation values selected")
np.savetxt('valid_nan_mask.txt', valid_mask)



test_mask = np.zeros((data.shape[0], data.shape[1]))
test_rows, test_cols = zip(*test_index_tuples)
print(len(test_rows), len(test_cols))
test_mask[test_rows, test_cols] = np.nan

print(f"Varifying: There are {int(np.sum(np.nan_to_num(test_mask, nan=1.0)))} test values selected")
np.savetxt('test_nan_mask.txt', test_mask)
