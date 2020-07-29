import json
import os
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='Dataset GT split')
parser.add_argument("--dataset", required=True, type=str, nargs='?')
args = parser.parse_args()

results_path = "../../Performance/Character Recognition/gt_{}.txt".format(args.dataset)
if not os.path.exists(results_path):
    os.makedirs(results_path)

with open("model_data/data_{}.json".format("val")) as f:
    text = json.load(f)

license_plate = {}
for tmp_line in open('../../Datasets/training_LP.txt', 'r'):
    tmp_line = tmp_line.strip().split('/')
    license_plate[tmp_line[0]] = tmp_line[1]

line = ""
for elem in text:
    filename = filename.split()[-1]
    print(filename)
    line += filename + "/" + license_plate[filename] + "\n"
 
with open(results_path , 'w') as f:
    f.write(line)