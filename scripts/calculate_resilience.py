import pandas as pd
import numpy as np
import argparse
import os

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-label-path", type=str, help="the path to true label")
    parser.add_argument("--predict-label-path", type=str, help="the path to predicted label")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print(args)
    true_label_path = args.true_label_path
    predict_label_path = args.predict_label_path


    raw_label = pd.read_csv(true_label_path)
    label = raw_label[['node', 'import_resilience']]
    predict = np.genfromtxt(predict_label_path, delimiter=',')


    diff_label = label.copy()
    diff_label['import_resilience'] = np.array(predict) - np.array(label['import_resilience'])
    # the stats of the difference
    print("The stats of the difference")
    print(diff_label.describe())
    diff_label['import_resilience'] = np.abs(diff_label['import_resilience'])
    print("The stats of the absolute difference")
    print(diff_label.describe())

    

 