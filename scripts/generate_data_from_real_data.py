import pandas as pd
import numpy as np
import os
import argparse

def remove_rows(df, n):
    if n > len(df):
        raise ValueError("n cannot be greater than the number of rows in the dataset.")
    else:
        drop_indices = np.random.choice(df.index, size=n, replace=False)
        df = df.drop(drop_indices)
    df.reset_index(drop=True, inplace=True)
    return df

def add_rows(df, n):
    new_rows = []
    existing_rows = set(tuple(x) for x in df[['Origin', 'Destination', 'COMM_TTL']].values)

    for _ in range(n):
        # Select random origin, destination and COMM_TTL
        origin = df['Origin'].sample().values[0]
        destination = df['Destination'].sample().values[0]
        comm_ttl = np.random.randint(1, 9)

        # Make sure there is no row with the same [Origin, Destination, COMM_TTL]
        while (origin, destination, comm_ttl) in existing_rows:
            origin = df['Origin'].sample().values[0]
            destination = df['Destination'].sample().values[0]
            comm_ttl = np.random.randint(1, 8)

        # Sample VAL, TON, AVGMILE from their distributions
        val = df['VAL'].sample().values[0]
        ton = df['TON'].sample().values[0]
        avgmile = df['AVGMILE'].sample().values[0]

        new_rows.append([comm_ttl, val, ton, avgmile, origin, destination])
        existing_rows.add((origin, destination, comm_ttl))

    df_new = pd.DataFrame(new_rows, columns=df.columns)
    df = pd.concat([df, df_new])
    df.reset_index(drop=True, inplace=True)
    return df

def change_entries(df, n):
    for _ in range(n):
        # Select random row
        idx = np.random.randint(0, len(df))

        # Select random column ('VAL', 'TON', or 'AVGMILE')
        col = np.random.choice(['VAL', 'TON', 'AVGMILE'])

        # Add noise (20 % of the range)
        noise = np.random.uniform(-0.1, 0.1) * (df[col].max() - df[col].min())
        new_value = df.loc[idx, col].item() + noise

        # Get the minimum value for the same origin and destination
        origin = df.loc[idx, 'Origin']
        destination = df.loc[idx, 'Destination']
        min_avgmile = df[(df['Origin'] == origin) & (df['Destination'] == destination)]['AVGMILE'].min()

        # Make sure the new 'AVGMILE' is not smaller than the minimum 'AVGMILE' for the same origin and destination
        df.loc[idx, col] = max(new_value, min_avgmile)
    df.reset_index(drop=True, inplace=True)
    return df

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="the path to the real data")
    parser.add_argument("--save-dir", type=str, help="the directory for saving the generated data")
    parser.add_argument("--size", type=int, help="the number of generated data")
    parser.add_argument("--noise-rate", type=float, help="the rate of noise")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    save_dir = args.save_dir
    size = args.size
    noise_rate = args.noise_rate

    original_df = pd.read_csv(data_path)
    num_of_modification = int(len(original_df) * noise_rate / 3.0)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for index in range(1, size + 1):
        df = remove_rows(original_df, num_of_modification)
        df = add_rows(df, num_of_modification)
        df = change_entries(df, num_of_modification)
        df.to_csv(os.path.join(save_dir, 'data_{}.csv'.format(index)), index=False)