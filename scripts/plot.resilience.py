import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import os

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-label-path", type=str, help="the path to true label")
    parser.add_argument("--predict-label-path", type=str, help="the path to predicted label")
    parser.add_argument("--info-dir", type=str, help="the directory for map info")
    parser.add_argument("--output-dir", type=str, help="the directory for saving the heatmap")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print(args)
    true_label_path = args.true_label_path
    predict_label_path = args.predict_label_path
    info_dir = args.info_dir
    output_dir = args.output_dir


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = gpd.read_file("zip://{}tl_rd22_us_state.zip".format(info_dir))

    raw_label = pd.read_csv(true_label_path)
    label = raw_label[['node', 'import_resilience']]
    predict = np.genfromtxt(predict_label_path, delimiter=',')


    merged_df = df.merge(label, left_on='STUSPS', right_on='node')  
    contiguous_us = merged_df[~merged_df['STUSPS'].isin(['AK', 'HI'])]
    # Plot the heatmap
    fig, ax = plt.subplots(1, figsize=(15, 8))
    contiguous_us.plot(column='import_resilience', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='black', legend=True)

    # Add title and axis labels
    plt.title('Contiguous US States Resilience Heatmap (Ground Truth)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, "true.png"), dpi=300)
    plt.close()


    predict_label = label.copy()
    predict_label['import_resilience'] = np.array(predict)
    merged_df = df.merge(predict_label, left_on='STUSPS', right_on='node')
    contiguous_us = merged_df[~merged_df['STUSPS'].isin(['AK', 'HI'])]
    # Plot the heatmap
    fig, ax = plt.subplots(1, figsize=(15, 8))
    contiguous_us.plot(column='import_resilience', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='black', legend=True)

    # Add title and axis labels
    plt.title('Contiguous US States Resilience Heatmap (Prediction)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, "pred.png"), dpi=300)
    plt.close()


    diff_label = label.copy()
    diff_label['import_resilience'] = np.array(predict) - np.array(label['import_resilience'])
    norm = colors.Normalize()
    merged_df = df.merge(diff_label, left_on='STUSPS', right_on='node')
    contiguous_us = merged_df[~merged_df['STUSPS'].isin(['AK', 'HI'])]
    # Plot the heatmap
    fig, ax = plt.subplots(1, figsize=(15, 8))
    norm = colors.Normalize(vmin=-0.4, vmax=0.4)
    contiguous_us.plot(column='import_resilience', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='black', legend=True, norm=norm)
    # Add title and axis labels
    plt.title('Contiguous US States Resilience Heatmap (Prediction - Ground Truth)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(os.path.join(output_dir, "diff.png"), dpi=300)
    plt.close()
    

 