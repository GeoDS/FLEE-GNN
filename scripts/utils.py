def create_edge_attr_index(df):
    df1 = df[["COMM_TTL", "VAL", "TON", "AVGMILE", "Origin", "Destination"]]
    # First, create a MultiIndex for the new columns using 'Commodity Category'
    df1 = df1.set_index(["COMM_TTL", 'Origin', 'Destination'])

    # Unstack the MultiIndex to create a wide format dataframe
    df2 = df1.unstack(level="COMM_TTL")
    # Flatten the MultiIndex columns

    # Reset the index to obtain 'Origin' and 'Destination' as columns
    df2.reset_index(inplace=True)

    df2.sort_index(axis=1, inplace=True)

    # Fill NaN values with 0
    df2.fillna(0, inplace=True)

     # Ensure all columns for 'COMM_TTL' are present
    # for i in range(1, 9):
    #     for col_name in ['VAL', 'TON', 'AVGMILE']:
    #         if (col_name, i) not in df2.columns:
    #             df2[(col_name, i)] = 0

    # Create a separate dataframe for edge_attr without 'Origin' and 'Destination' columns
    edge_attr = df2.drop(columns=['Origin', 'Destination'])

    # Create a separate dataframe for edge_attr with 'Origin' and 'Destination' columns
    edge_index = df2[['Origin', 'Destination']]

    return edge_attr, edge_index


def normalize_features(features):
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    return (features - means) / stds