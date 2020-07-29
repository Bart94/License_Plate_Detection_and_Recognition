import os, sys, argparse
from glob import glob
from difflib import SequenceMatcher

def load_plates(path):
    pgt={} 
    f = open(path)
    for line in f:
        line =[x.strip() for x in line.split('/')]
        k = line[0]
        v = line[1]
        pgt[k]=v
    return pgt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recognition Performance')
    parser.add_argument("--gt_file", type=str, nargs='?', default='training_LP.txt', help="File con tutte le targhe corrette.")
    parser.add_argument("--result_file", type=str, nargs='?', default='result_LP.txt', help="File con tutte le targhe classificate.")
    args = parser.parse_args()

    gt = load_plates(args.gt_file)
    result = load_plates(args.result_file)

    total_num_samples = len(gt.keys())
    detected_plates = 0
    recognized_plates = 0
    cumulative_similarity = 0.0
    for key in result.keys():
        detected_plates += 1
        similarity = SequenceMatcher(None, gt[key], result[key]).ratio()
        cumulative_similarity += similarity
        if similarity == 1.0:
            recognized_plates += 1

    LPRA = recognized_plates/detected_plates
    OA = recognized_plates/total_num_samples
    ALPS = cumulative_similarity/detected_plates

    print('LPRA=%f OA=%f ALPS=%f'%(LPRA, OA, ALPS))