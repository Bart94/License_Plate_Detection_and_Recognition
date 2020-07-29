import json
import os
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='Dataset GT split')
parser.add_argument("--dataset", required=True, type=str, nargs='?')
args = parser.parse_args()

results_path = "../gt_{}".format(args.dataset)
if not os.path.exists(results_path):
    os.makedirs(results_path)

with open("../data_{}.json".format("val")) as f:
  text = json.load(f)

for elem in text:
  filename = "../{}.txt".format(elem)
  print(filename)
  copyfile(filename, "{}/{}.txt".format(results_path,os.path.basename(elem)))