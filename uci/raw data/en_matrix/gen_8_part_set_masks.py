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
rows_repeated = [i for i in range(data.shape[0]) 
                   for _ in range(5, data.shape[1])]
cols_repeated = [j for _ in range(data.shape[0]) 
                   for j in range(5, data.shape[1])]
print(len(rows_repeated), len(cols_repeated))
assert(len(cols_repeated) == len(rows_repeated))
index_tuples_lst = list(zip(rows_repeated, cols_repeated))
print(len(set(index_tuples_lst)))
assert(len(cols_repeated) == len(set(index_tuples_lst)))
assert(len(rows_repeated) == len(set(index_tuples_lst)))
print(len(index_tuples_lst), len(set(index_tuples_lst)))
random.shuffle(index_tuples_lst)

sets = []

sets.append(index_tuples_lst[:int(np.round(len(index_tuples_lst) * 0.125))])
sets.append(index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.125)): int(np.round(len(index_tuples_lst) * 0.25))])
sets.append(index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.25)): int(np.round(len(index_tuples_lst) * 0.375))])
sets.append(index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.375)): int(np.round(len(index_tuples_lst) * 0.5))])
sets.append(index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.5)): int(np.round(len(index_tuples_lst) * 0.625))])
sets.append(index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.625)): int(np.round(len(index_tuples_lst) * 0.75))])
sets.append(index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.75)): int(np.round(len(index_tuples_lst) * 0.875))])
sets.append(index_tuples_lst[int(np.round(len(index_tuples_lst) * 0.875)):])

print(f"Number of items on each set in order are: {len(sets[0])}, {len(sets[1])},"
      f" {len(sets[2])}, {len(sets[3])}, {len(sets[4])}, {len(sets[5])}, {len(sets[6])}, {len(sets[7])}")

for set_index, a_set in enumerate(sets):
    mask = np.zeros((data.shape[0], data.shape[1]))
    rows, cols = zip(*a_set)
    print(len(rows), len(cols))
    mask[rows, cols] = np.nan
    print(f"Varifying: There are {int(np.sum(np.nan_to_num(mask, nan=1.0)))} values selected on set {set_index + 1}")
    np.savetxt(f'set_{1 + set_index}_of_8_set_mask.txt', mask)


