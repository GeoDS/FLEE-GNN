import pandas as pd
import numpy as np
import os
import argparse
import random
from itertools import product

def generate_random_dataset(num_rows):
    us_state_abbrevs = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
    ]

    def random_state_weights(states):
        weights = [random.randint(1, 10) for state in states]
        return np.array(weights) / sum(weights)

    state_weights = random_state_weights(us_state_abbrevs)
    
    unique_trip_combinations = list(product(us_state_abbrevs, us_state_abbrevs, range(1, 9)))
    random.shuffle(unique_trip_combinations)
    # Calculate the weights for each combination
    combination_weights = [state_weights[us_state_abbrevs.index(o)] * state_weights[us_state_abbrevs.index(d)] for o, d, _ in unique_trip_combinations]
    
    unique_trip_combinations = random.choices(unique_trip_combinations, k=num_rows, weights=combination_weights)
    unique_trip_combinations = set(unique_trip_combinations)
    
    data = []
    for origin, destination, comm_ttl in unique_trip_combinations:
        val = ton = avgmil = 0
        while val == 0 or ton == 0 or avgmil == 0:
            val_mean = random.uniform(300, 600)
            val_std = random.uniform(1000, 2000)
            val = max(0, np.random.normal(val_mean, val_std))

            ton_mean = random.uniform(300, 600)
            ton_std = random.uniform(1500, 2500)
            ton = max(0, np.random.normal(ton_mean, ton_std))

            avgmil_mean = random.uniform(500, 1000)
            avgmil_std = random.uniform(500, 1500)
            avgmil = max(0, np.random.normal(avgmil_mean, avgmil_std))

        data.append([origin, destination, val, ton, avgmil, comm_ttl])
    
    df = pd.DataFrame(data, columns=['Origin', 'Destination', 'VAL', 'TON', 'AVGMILE', 'COMM_TTL'])
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, help="the directory for saving the generated data")
    parser.add_argument("--size", type=int, help="the number of generated data")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    size = args.size
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for index in range(1, size+1):
        random_dataset = generate_random_dataset(random.randint(1000, 2000))
        random_dataset.to_csv(os.path.join(save_dir, 'data_{}.csv'.format(index)), index=False)