import numpy as np
import pandas as pd
import shapely

def reconstruct_degree_sequences(dataframe):
    degree_sequence_1 = []
    degree_sequence_2 = []
    for row in np.arange(1, dataframe.shape[0]):
        for column in np.arange(1, dataframe.shape[1]):
            degree_1 = int(dataframe.iloc[row, 0])
            degree_2 = int(dataframe.iloc[0, column])
            count = int(dataframe.iloc[row, column])
            degree_sequence_1.extend([degree_1] * count)
            degree_sequence_2.extend([degree_2] * count)
    degree_sequence_1 = np.array(degree_sequence_1)
    degree_sequence_2 = np.array(degree_sequence_2)
    return degree_sequence_1, degree_sequence_2

def sample_synthetic_locations(size, uniform_proportion, cluster_proportions, cluster_spreads, square_size=1/np.sqrt(2), margin=0):
    uniform_size = int(size * uniform_proportion)
    cluster_sizes = (np.array(cluster_proportions) * size).astype(int)
    remainder = size - uniform_size - sum(cluster_sizes)
    if remainder > 0:
        indices = np.random.choice(len(cluster_sizes), size=remainder, replace=True)
        for i in indices:
            cluster_sizes[i] += 1
    uniform = np.random.uniform(0, square_size, size=(uniform_size, 2))
    clustered = []
    centers = np.random.uniform(margin, square_size - margin, size=(len(cluster_sizes), 2))
    for i, (size_i, spread_i) in enumerate(zip(cluster_sizes, cluster_spreads)):
        points = []
        while len(points) < size_i:
            candidate = np.random.normal(loc=centers[i], scale=spread_i, size=(size_i - len(points), 2))
            valid = candidate[(candidate >= 0).all(axis=1) & (candidate <= square_size).all(axis=1)]
            points.extend(valid.tolist())
        clustered.append(np.array(points[:size_i]))
    clustered = np.vstack(clustered)
    locations = np.vstack([clustered, uniform])
    return locations

def sample_geographic_locations(size, dataframe):
    probabilities = dataframe["population"] / np.sum(dataframe["population"])
    sampled_blocks = np.random.choice(dataframe.index, size=size, p=probabilities)
    sampled_points = []
    for index in sampled_blocks:
        block = dataframe.loc[index, "geometry"]
        minx, miny, maxx, maxy = block.bounds
        while True:
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            if shapely.geometry.Point(x, y).within(block):
                sampled_points.append((x, y))
                break
    locations = np.array(sampled_points)
    return locations

def proportion_of_dropped_edges(dataframe):
    E1 = dataframe["number_of_net1_edges_global_pre"]
    E2 = dataframe["number_of_net2_edges_global_pre"]
    L1 = dataframe["number_of_net1_edges_global_post"]
    L2 = dataframe["number_of_net2_edges_global_post"]
    dataframe["proportion_of_net1_dropped_edges"] = (E1 - L1) / E1
    dataframe["proportion_of_net2_dropped_edges"] = (E2 - L2) / E2
    return dataframe

def number_of_overlap_edges_to_number_of_net1_edges(dataframe):
    dataframe["number_of_overlap_edges_to_number_of_net1_edges"] = (
        dataframe["number_of_overlap_edges_global_post"] /
        dataframe["number_of_net1_edges_global_post"]
    )
    return dataframe

def number_of_overlap_edges_to_number_of_net2_edges(dataframe):
    dataframe["number_of_overlap_edges_to_number_of_net2_edges"] = (
        dataframe["number_of_overlap_edges_global_post"] /
        dataframe["number_of_net2_edges_global_post"]
    )
    return dataframe

def normalized_global_jaccard(dataframe):
    L1 = dataframe["number_of_net1_edges_global_post"]
    L2 = dataframe["number_of_net2_edges_global_post"]
    L1U2 = dataframe["number_of_union_edges_global_post"]
    L1I2 = dataframe["number_of_overlap_edges_global_post"]
    dataframe["normalized_global_jaccard"] = (L1I2 / L1U2) / (np.minimum(L1, L2) / np.maximum(L1, L2))
    return dataframe

def joint_deg_mean_number_of_overlap_edges(dataframe):
    dataframe["joint_deg_mean_number_of_overlap_edges"] = (
        dataframe["joint_deg_number_of_overlap_edges_post"] /
        dataframe["joint_deg_number_of_nodes_post"].replace(0, np.nan)
    )
    return dataframe

def joint_deg_mean_number_of_overlap_edges_to_number_of_net1_edges(dataframe):
    dataframe["joint_deg_mean_number_of_overlap_edges_to_number_of_net1_edges"] = (
        dataframe["joint_deg_mean_number_of_overlap_edges"] /
        dataframe["deg1"].replace(0, np.nan)
    )
    return dataframe

def joint_deg_mean_number_of_overlap_edges_to_number_of_net2_edges(dataframe):
    dataframe["joint_deg_mean_number_of_overlap_edges_to_number_of_net2_edges"] = (
        dataframe["joint_deg_mean_number_of_overlap_edges"] /
        dataframe["deg2"].replace(0, np.nan)
    )
    return dataframe

def joint_deg_mean_normalized_local_jaccard(dataframe):
    dataframe["joint_deg_mean_normalized_local_jaccard"] = (
        dataframe["joint_deg_normalized_local_jaccard_sum_post"] /
        dataframe["joint_deg_number_of_nodes_post"].replace(0, np.nan)
    )
    return dataframe

def joint_distributions(dataframe):
    distributions = dataframe[["unique_id", "sample_id", "model_id", "joint_deg_distributions"]]
    distributions = distributions.explode("joint_deg_distributions")
    identifiers = distributions.drop(columns="joint_deg_distributions").reset_index(drop=True)
    dictionaries = pd.json_normalize(distributions["joint_deg_distributions"])
    distributions = pd.concat([identifiers, dictionaries], axis=1)
    distributions = distributions.groupby(
        ["unique_id", "model_id", "deg1", "deg2"], as_index=False
    ).agg({
        "joint_deg_number_of_nodes_pre": "sum",
        "joint_deg_number_of_nodes_post": "sum",
        "joint_deg_number_of_overlap_edges_post": "sum",
        "joint_deg_normalized_local_jaccard_sum_post": "sum"
    })
    distributions = distributions.astype({
        "unique_id": "uint16",
        "deg1": "uint16",
        "deg2": "uint16",
        "joint_deg_number_of_nodes_pre": "uint32",
        "joint_deg_number_of_nodes_post": "uint32",
        "joint_deg_number_of_overlap_edges_post": "uint32",
        "joint_deg_normalized_local_jaccard_sum_post": "float32"
    })
    return distributions

def get_unique_ids(metadata, net_size=None, deg_seq_corr=None, char_dist=None, homophily=None):
    df = metadata.copy()
    if net_size is not None:
        df = df[df["net_size"] == net_size]
    if deg_seq_corr is not None:
        df = df[df["deg_seq_corr"] == deg_seq_corr]
    if char_dist is not None:
        df = df[df["char_dist"] == char_dist]
    if homophily is not None:
        df = df[df["homophily"] == homophily]
    return df["unique_id"].unique()

def create_table(dataframe, metric, row_param, col_param):
    blocks = []
    for model in dataframe['model_id'].unique():
        subdf = dataframe[dataframe['model_id'] == model]
        grouped = subdf.groupby([row_param, col_param])[metric].agg(['mean', 'std']).round(3)
        formatted = grouped.apply(lambda x: f"{x['mean']} ± {x['std']}", axis=1)
        grid = formatted.unstack(col_param)
        grid = grid.sort_index(ascending=False)
        header = pd.DataFrame([[f"{row_param} / {col_param}"] + list(grid.columns)], columns=[grid.index.name] + list(grid.columns))
        grid = grid.reset_index()
        block = pd.concat([header, grid], ignore_index=True)
        model_row = pd.DataFrame([[f"{model}"] + [""] * (block.shape[1] - 1)], columns=block.columns)
        block = pd.concat([model_row, block], ignore_index=True)
        blocks.append(block)
    return blocks

def export_tables(dataframe, configuration, filename):
    with pd.ExcelWriter(filename) as writer:
        for entry in configuration:
            metric = entry["metric"]
            row_param, col_param = entry["params"]
            sheet_name = entry["sheet"]
            blocks = create_table(dataframe, metric, row_param, col_param)
            stacked = pd.concat(blocks, ignore_index=True)
            stacked.to_excel(writer, sheet_name=sheet_name, index=False, header=False)