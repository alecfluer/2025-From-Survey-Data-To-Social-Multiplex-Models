import networkx as nx
import numpy as np
import scipy
import sdcdp

def worker_for_sda_prob(parameters):
    net_size, char_dist, homophily = parameters["sda_prob_params"]
    networks_dataframe = parameters["networks_dataframe"]
    features_dataframe = parameters["features_dataframe"]

    MultiplexSDA = sdcdp.sda.MultiplexSDA()
    MultiplexSDA.networks_dataframe = networks_dataframe
    MultiplexSDA.features_dataframe = features_dataframe
    probabilities = MultiplexSDA.compute_prob_matrices()

    result = {}
    result["sda_prob_params"] = (net_size, char_dist, homophily)
    result["probabilities"] = probabilities

    return result

def worker_for_deg_seqs(parameters):
    unique_id = parameters["unique_id"]
    sample_id = parameters["sample_id"]
    network_names = parameters["network_names"]
    net_size, m1, v1, m2, v2, deg_seq_corr = parameters["deg_seqs_params"]

    omega = np.full((2, 2), deg_seq_corr)
    np.fill_diagonal(omega, 1)
    mean_deg_seq = sdcdp.seq.sample_correlated_mean_degree_sequences(means=[m1, m2], variances=[v1, v2], size=net_size, omega=omega)
    net1_deg_seq = sdcdp.seq.sample_degree_sequence(mean_degree_sequence=mean_deg_seq[0])
    net2_deg_seq = sdcdp.seq.sample_degree_sequence(mean_degree_sequence=mean_deg_seq[1])
    degree_sequences = {network_names[0]: net1_deg_seq, network_names[1]: net2_deg_seq}

    result = {}
    result["unique_id"] = unique_id
    result["sample_id"] = sample_id
    result["degree_sequences"] = degree_sequences

    return result

def worker_for_networks(parameters):
    unique_id = parameters["unique_id"]
    sample_id = parameters["sample_id"]
    network_names = parameters["network_names"]
    probabilities = parameters["probabilities"]
    degree_sequences = parameters["degree_sequences"]
    compare_models = parameters["compare_models"]
    compute_metrics = parameters["compute_metrics"]

    results = []
    for model_id, model_function in compare_models.items():
        networks = {}
        for network_name in network_names:
            networks[network_name] = model_function(
                degree_sequence=degree_sequences[network_name],
                probabilities=probabilities[network_name]
            )
        
        metrics = compute_metrics(networks)
        results.append({
            "unique_id": unique_id,
            "sample_id": sample_id,
            "model_id": model_id,
            **metrics
        })
    
    return results

def metrics_undirected(networks):

    metrics = {}
    joint_deg_distributions = []

    net_names = list(networks.keys())
    net1_pre = networks[net_names[0]]
    net2_pre = networks[net_names[1]]

    metrics["number_of_net1_edges_global_pre"] = len(net1_pre.edges())
    metrics["number_of_net2_edges_global_pre"] = len(net2_pre.edges())

    net1_post = nx.Graph(net1_pre)
    net2_post = nx.Graph(net2_pre)
    net1_post.remove_edges_from(nx.selfloop_edges(net1_post))
    net2_post.remove_edges_from(nx.selfloop_edges(net2_post))

    metrics["number_of_net1_edges_global_post"] = len(net1_post.edges())
    metrics["number_of_net2_edges_global_post"] = len(net2_post.edges())

    E1 = set(net1_post.edges())
    E2 = set(net2_post.edges())

    metrics["number_of_union_edges_global_post"] = len(E1 | E2)
    metrics["number_of_overlap_edges_global_post"] = len(E1 & E2)

    net1_post_deg_seq = np.array(list(dict(net1_post.degree()).values()))
    net2_post_deg_seq = np.array(list(dict(net2_post.degree()).values()))

    metrics["pearson_corr_global_post"] = scipy.stats.pearsonr(net1_post_deg_seq, net2_post_deg_seq)[0]
    metrics["spearman_corr_global_post"] = scipy.stats.spearmanr(net1_post_deg_seq, net2_post_deg_seq)[0]

    keys = set()
    joint_deg_number_of_nodes_pre = {}
    joint_deg_number_of_nodes_post = {}
    joint_deg_number_of_overlap_edges_post = {}
    joint_deg_normalized_local_jaccard_sum_post = {}

    for node in net1_pre.nodes():
        d1 = net1_pre.degree(node)
        d2 = net2_pre.degree(node)
        key = (d1, d2)

        if key not in keys:
            keys.add(key)
            joint_deg_number_of_nodes_pre[key] = 0
            joint_deg_number_of_nodes_post[key] = 0
            joint_deg_number_of_overlap_edges_post[key] = 0
            joint_deg_normalized_local_jaccard_sum_post[key] = 0

        joint_deg_number_of_nodes_pre[key] += 1

        d1 = net1_post.degree(node)
        d2 = net2_post.degree(node)
        key = (d1, d2)

        if key not in keys:
            keys.add(key)
            joint_deg_number_of_nodes_pre[key] = 0
            joint_deg_number_of_nodes_post[key] = 0
            joint_deg_number_of_overlap_edges_post[key] = 0
            joint_deg_normalized_local_jaccard_sum_post[key] = 0

        joint_deg_number_of_nodes_post[key] += 1

        N1 = set(net1_post.neighbors(node))
        N2 = set(net2_post.neighbors(node))
        L1, L2 = len(N1), len(N2)
        L1U2 = len(N1 | N2)
        L1I2 = len(N1 & N2)

        if L1I2 > 0:
            joint_deg_normalized_local_jaccard_sum_post[key] += (L1I2 / L1U2) / (min(L1, L2) / max(L1, L2))

    for node1, node2 in E1 & E2:
        for node in [node1, node2]:
            d1 = net1_post.degree(node)
            d2 = net2_post.degree(node)
            key = (d1, d2)

            joint_deg_number_of_overlap_edges_post[key] += 1

    for (d1, d2) in keys:
        joint_deg_distributions.append({
            "deg1": d1,
            "deg2": d2,
            "joint_deg_number_of_nodes_pre": joint_deg_number_of_nodes_pre[(d1, d2)],
            "joint_deg_number_of_nodes_post": joint_deg_number_of_nodes_post[(d1, d2)],
            "joint_deg_number_of_overlap_edges_post": joint_deg_number_of_overlap_edges_post[(d1, d2)],
            "joint_deg_normalized_local_jaccard_sum_post": joint_deg_normalized_local_jaccard_sum_post[(d1, d2)]
        })

    metrics["joint_deg_distributions"] = joint_deg_distributions

    return metrics