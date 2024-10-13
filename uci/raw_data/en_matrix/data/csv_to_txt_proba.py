import glob
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--csv_data_file_name', type=str, default='Provide CSV data file name path') 
args = parser.parse_args()
print(args)

data = []
e_cause = []

with open(glob.glob(args.csv_data_file_name)[0]) as fl:
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
np.savetxt('data.txt', data)
print(e_cause_dict)