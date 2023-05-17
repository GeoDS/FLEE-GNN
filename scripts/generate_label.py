import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import json
import argparse
import os

def calculate_node_resilience(edges, comm_level_dict, adjacency_dict, alpha=1.0, beta=0.9, level='state'):
    comm_stat_dict = {'comm_ttl': {},
                      'l1_1_val_s': 0.0,
                      'l1_2_val_s': 0.0,
                      'l1_val_s': 0.0,
                     }

    for e in edges:
        origin, destination, edge_data = e
        # initialize stat dict for specific comm_ttl
        if edge_data['COMM_TTL'] not in comm_stat_dict['comm_ttl']:
            comm_stat_dict['comm_ttl'][edge_data['COMM_TTL']] = {}
            comm_stat_dict['comm_ttl'][edge_data['COMM_TTL']]['VAL'] = []
        if level=='state': # for state-level data, we use state adjacency (unfixed beta)
            VAL_calibrated = edge_data['VAL'] * (1.0 if edge_data['AVGMILE'] < alpha else edge_data['AVGMILE']**0.5) * (beta if origin in adjacency_dict[destination] else 1.0)
        else: # for region-level and division-level data, we do not use adjacency (fixed beta)
            VAL_calibrated = edge_data['VAL'] * (1.0 if edge_data['AVGMILE'] < alpha else edge_data['AVGMILE']**0.5) * beta
        comm_stat_dict['comm_ttl'][edge_data['COMM_TTL']]['VAL'].append(VAL_calibrated)
        if edge_data['COMM_TTL'] in agri_comm_level_dict['l1_1']:
            comm_stat_dict['l1_1_val_s'] += VAL_calibrated
        if edge_data['COMM_TTL'] in agri_comm_level_dict['l1_2']:
            comm_stat_dict['l1_2_val_s'] += VAL_calibrated

    comm_stat_dict['l1_val_s'] = comm_stat_dict['l1_1_val_s'] + comm_stat_dict['l1_2_val_s']
    
    # calculate dependency risk for level 1, set l1_1
    for comm_ttl in agri_comm_level_dict['l1_1']:
        if 'l1_1_sum_weighted_dependency_risk' not in comm_stat_dict:
            comm_stat_dict['l1_1_sum_weighted_dependency_risk'] = 0.
        if 'l1_1_sum_dependency_risk' not in comm_stat_dict:
            comm_stat_dict['l1_1_sum_dependency_risk'] = 0.
        if comm_ttl not in comm_stat_dict['comm_ttl']:
            continue
        v_sum = np.sum(comm_stat_dict['comm_ttl'][comm_ttl]['VAL'])
        comm_stat_dict['comm_ttl'][comm_ttl]['dependency_risk'] = np.prod([(v/v_sum)**(v/v_sum) for v in comm_stat_dict['comm_ttl'][comm_ttl]['VAL']])
        comm_stat_dict['l1_1_sum_dependency_risk'] += comm_stat_dict['comm_ttl'][comm_ttl]['dependency_risk']
        comm_stat_dict['comm_ttl'][comm_ttl]['weighted_dependency_risk'] = comm_stat_dict['comm_ttl'][comm_ttl]['dependency_risk'] * v_sum
        comm_stat_dict['l1_1_sum_weighted_dependency_risk'] += comm_stat_dict['comm_ttl'][comm_ttl]['weighted_dependency_risk']
    
    # calculate dependency risk for level 1, set l1_2
    for comm_ttl in agri_comm_level_dict['l1_2']:
        if 'l1_2_sum_weighted_dependency_risk' not in comm_stat_dict:
            comm_stat_dict['l1_2_sum_weighted_dependency_risk'] = 0.
        if 'l1_2_sum_dependency_risk' not in comm_stat_dict:
            comm_stat_dict['l1_2_sum_dependency_risk'] = 0.
        if comm_ttl not in comm_stat_dict['comm_ttl']:
            continue
        v_sum = np.sum(comm_stat_dict['comm_ttl'][comm_ttl]['VAL'])
        comm_stat_dict['comm_ttl'][comm_ttl]['dependency_risk'] = np.prod([(v/v_sum)**(v/v_sum) for v in comm_stat_dict['comm_ttl'][comm_ttl]['VAL']])
        comm_stat_dict['l1_2_sum_dependency_risk'] += comm_stat_dict['comm_ttl'][comm_ttl]['dependency_risk']
        comm_stat_dict['comm_ttl'][comm_ttl]['weighted_dependency_risk'] = comm_stat_dict['comm_ttl'][comm_ttl]['dependency_risk'] * v_sum
        comm_stat_dict['l1_2_sum_weighted_dependency_risk'] += comm_stat_dict['comm_ttl'][comm_ttl]['weighted_dependency_risk']
        
    comm_stat_dict['l1_1_dependency_risk'] = np.prod([(comm_stat_dict['comm_ttl'][l1_1_comm]['weighted_dependency_risk'] /
                                                       comm_stat_dict['l1_1_sum_weighted_dependency_risk'])**
                                                      (comm_stat_dict['comm_ttl'][l1_1_comm]['weighted_dependency_risk'] /
                                                       comm_stat_dict['l1_1_sum_weighted_dependency_risk']) \
                                                      for l1_1_comm in agri_comm_level_dict['l1_1'] \
                                                      if l1_1_comm in comm_stat_dict['comm_ttl']])
    
    comm_stat_dict['l1_2_dependency_risk'] = np.prod([(comm_stat_dict['comm_ttl'][l1_2_comm]['weighted_dependency_risk'] /
                                                       comm_stat_dict['l1_2_sum_weighted_dependency_risk'])**
                                                      (comm_stat_dict['comm_ttl'][l1_2_comm]['weighted_dependency_risk'] /
                                                       comm_stat_dict['l1_2_sum_weighted_dependency_risk']) \
                                                      for l1_2_comm in agri_comm_level_dict['l1_2'] \
                                                      if l1_2_comm in comm_stat_dict['comm_ttl']])
    
    # weighting
    comm_stat_dict['l1_1_abs_dependency_risk'] = comm_stat_dict['l1_1_dependency_risk'] * comm_stat_dict['l1_1_sum_weighted_dependency_risk']
    comm_stat_dict['l1_2_abs_dependency_risk'] = comm_stat_dict['l1_2_dependency_risk'] * comm_stat_dict['l1_2_sum_weighted_dependency_risk']
    # normalization
    comm_stat_dict['l1_1_norm_abs_dependency_risk'] = comm_stat_dict['l1_1_abs_dependency_risk'] / (comm_stat_dict['l1_1_abs_dependency_risk'] + comm_stat_dict['l1_2_abs_dependency_risk'])
    comm_stat_dict['l1_2_norm_abs_dependency_risk'] = comm_stat_dict['l1_2_abs_dependency_risk'] / (comm_stat_dict['l1_1_abs_dependency_risk'] + comm_stat_dict['l1_2_abs_dependency_risk']) 
    
    # dependence risk for level 1
    comm_stat_dict['l1_dependency_risk'] = np.prod([comm_stat_dict['l1_1_norm_abs_dependency_risk']**comm_stat_dict['l1_1_norm_abs_dependency_risk'],
                                               comm_stat_dict['l1_2_norm_abs_dependency_risk']**comm_stat_dict['l1_2_norm_abs_dependency_risk']])
    
    # node resilience
    comm_stat_dict['node_resilience'] = 1 - comm_stat_dict['l1_dependency_risk'] * (comm_stat_dict['l1_1_abs_dependency_risk'] + comm_stat_dict['l1_2_abs_dependency_risk']) / comm_stat_dict['l1_val_s']
    
    return comm_stat_dict

def get_resilience_df(G, agri_comm_level_dict, adjacency_dict, alpha=1.0, beta=0.9, level='state'):
    node_list = []
    node_import_resilience_list = []
    node_export_resilience_list = []
    node_import_impact_to_network_list = []
    node_export_impact_to_network_list = []

    node_import_res_list = []
    node_export_res_list = []

    for node in tqdm(G.nodes):
        node_list.append(node)
        import_res = calculate_node_resilience(G.in_edges(node, data=True), agri_comm_level_dict, adjacency_dict, alpha, beta, level)
        export_res = calculate_node_resilience(G.out_edges(node, data=True), agri_comm_level_dict, adjacency_dict, alpha, beta, level)
        node_import_resilience_list.append(import_res['node_resilience'])
        node_export_resilience_list.append(export_res['node_resilience'])
        node_import_impact_to_network_list.append(import_res['node_resilience'] * import_res['l1_val_s'])
        node_export_impact_to_network_list.append(export_res['node_resilience'] * export_res['l1_val_s'])
        node_import_res_list.append(import_res)
        node_export_res_list.append(export_res)

    node_import_resilience_network_list = [impact/np.sum(node_import_impact_to_network_list) for impact in node_import_impact_to_network_list]
    node_export_resilience_network_list = [impact/np.sum(node_export_impact_to_network_list) for impact in node_export_impact_to_network_list]

    node_resilience_df = pd.DataFrame()
    node_resilience_df['node'] = node_list
    node_resilience_df['import_resilience'] = node_import_resilience_list
    node_resilience_df['export_resilience'] = node_export_resilience_list
    node_resilience_df['import_impact_to_network'] = node_import_impact_to_network_list
    node_resilience_df['export_impact_to_network'] = node_export_impact_to_network_list
    node_resilience_df['import_resilience_network'] = node_import_resilience_network_list
    node_resilience_df['export_resilience_network'] = node_export_resilience_network_list
    return node_resilience_df

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-dir", type=str, help="the directory for map info")
    parser.add_argument("--data-dir", type=str, help="the path to the graph data")
    parser.add_argument("--label-dir", type=str, help="the directory for saving the label")
    parser.add_argument("--size", type=int, help="the number of graph data")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    print(args)
    info_dir = args.info_dir
    data_dir = args.data_dir
    label_dir = args.label_dir
    size = args.size

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    with open(os.path.join(info_dir, 'state_a2f.json'), 'r') as f:
        state_a2f_dict = json.load(f)
    
    with open(os.path.join(info_dir, 'state_adjacency.json'), 'r') as f:
        state_adjacency_dict = json.load(f)
        
    with open(os.path.join(info_dir, 'state_f2latlon.json'), 'r') as f:
        state_f2latlon_list = json.load(f)
        
    state_f2a_dict = {}
    for key in state_a2f_dict:
        state_f2a_dict[state_a2f_dict[key]] = key    

    state_a2latlon_dict = {}
    for d in state_f2latlon_list:
        state_a2latlon_dict[state_f2a_dict[d['state']]] = {'lat': d["latitude"], 'lon': d["longitude"]}

    comm_dict = {
        1: 'Animals and fish (live)',
        2: 'Cereal grains (includes seed)',
        3: 'Agricultural products (excludes animal feed, cereal grains, and forage products)',
        4: 'Animal feed, eggs, honey, and other products of animal origin',
        5: 'Meat, poultry, fish, seafood, and their preparations',
        6: 'Milled grain products and preparations, and bakery products',
        7: 'Other prepared foodstuffs, and fats and oils (CFS10)',
        8: 'Alcoholic beverages and denatured alcohol (CFS20)'
    }

    agri_comm_level_dict = {}
    agri_comm_level_dict['l1_1'] = [comm_dict[comm] for comm in comm_dict if comm in [1,2,3,4,5]]
    agri_comm_level_dict['l1_2'] = [comm_dict[comm] for comm in comm_dict if comm in [6,7,8]]

    np.seterr(all='raise')
    failed_index = []
    for index in range(1,size + 1):
        df = pd.read_csv(os.path.join(data_dir, 'data_{}.csv'.format(index)))
        df['COMM_TTL'] = df['COMM_TTL'].apply(lambda code: comm_dict[code])
        df['GEO_TTL'] = df['Origin'].apply(lambda a: state_a2f_dict[a])
        df['DDESTGEO_TTL'] = df['Destination'].apply(lambda a: state_a2f_dict[a])
        df['DMODE_TTL'] = 'All modes'
        df['origin_lat'] = df['Origin'].apply(lambda abbr: state_a2latlon_dict[abbr]['lat'])
        df['destination_lat'] = df['Destination'].apply(lambda abbr: state_a2latlon_dict[abbr]['lat'])
        df['origin_lon'] = df['Origin'].apply(lambda abbr: state_a2latlon_dict[abbr]['lon'])
        df['destination_lon'] = df['Destination'].apply(lambda abbr: state_a2latlon_dict[abbr]['lon'])

        G = nx.from_pandas_edgelist(source = 'Origin', target = 'Destination', df = df, edge_attr=True, create_using=nx.MultiDiGraph())
        try:
            resilience = get_resilience_df(G, agri_comm_level_dict, state_adjacency_dict, alpha=1.0, beta=0.9, level='state')
            resilience.sort_values('import_resilience', ascending=False)
            resilience.to_csv(os.path.join(label_dir, 'label_{}.csv'.format(index)), index=False)
        except:
            print("index: {} failed".format(index))
            failed_index.append(index)
        
    print(failed_index)