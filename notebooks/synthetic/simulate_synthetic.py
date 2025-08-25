# %% [markdown]
# ## **Simulate** Synthetic

# %% [markdown]
# **Required Imports**

# %%
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd
import sdcdp

sys.path.append('../../')
import src

# %% [markdown]
# **Setup: Multiprocessing**

# %%
number_of_samples = 100
number_of_mp_cpus = 200

# %%
mp_worker_for_sda_prob = src.workers.worker_for_sda_prob
mp_worker_for_deg_seqs = src.workers.worker_for_deg_seqs
mp_worker_for_networks = src.workers.worker_for_networks

mp_params_for_sda_prob = []
mp_params_for_deg_seqs = []
mp_params_for_networks = []

unique_id = -1
metadata = []
locations = []

# %% [markdown]
# **Setup: Compare Models**

# %%
compare_models = {}
compare_models["CMS"] = src.models.configuration_undirected
compare_models["SDC"] = src.models.sdc_undirected
compare_models["SDC-DP"] = src.models.sdcdp_undirected

# %% [markdown]
# **Setup: Compute Metrics**

# %%
compute_metrics = src.workers.metrics_undirected

# %% [markdown]
# **Setup: Synthetic Parameters**

# %%
network_names = ["Network 1", "Network 2"]
feature_names = ["Cartesian"]

net_size_params = 500 * np.arange(1, 16, 1)

char_dist_params = [0.10, 0.20, 0.40, 0.80, 1.60]
homophily_params = [2, 4, 8, 16]

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

net1_deg_seq_params = [(5, 50)]
net2_deg_seq_params = [(10, 100)]
deg_seq_corr_params = [0.00, 0.25, 0.50, 0.75, 1.00]

# %% [markdown]
# **Multiprocessing: Initialize**

# %%
print("⏳ [Stage] Generating parameters for multiprocessing SDA probabilities and degree sequences...", flush=True)
t = time.time()

# %%
MultiplexSDA = sdcdp.sda.MultiplexSDA()
MultiplexSDA.add_networks_from(networks=network_names)
MultiplexSDA.add_features_from(features=feature_names)

for net_size in net_size_params:

    loc = src.data.sample_synthetic_locations(
        size=net_size,
        uniform_proportion=0.4,
        cluster_proportions=[0.04, 0.08, 0.16, 0.32],
        cluster_spreads=[0.02, 0.02, 0.04, 0.04],
        square_size=1/np.sqrt(2),
        margin=0.05
    )

    locations.append(loc)
    MultiplexSDA.clear_feature_params_from(features=feature_names)
    MultiplexSDA.assign_feature_params(feature="Cartesian", locations=loc, dist_func=euclidean_distance)
    
    for char_dist in char_dist_params:
        for homophily in homophily_params:

            MultiplexSDA.assign_network_params(network="Network 1", feature="Cartesian", char_dist=char_dist, homophily=homophily, weight=1)
            MultiplexSDA.assign_network_params(network="Network 2", feature="Cartesian", char_dist=char_dist, homophily=homophily, weight=1)
            
            networks_dataframe = MultiplexSDA.networks_dataframe.copy()
            features_dataframe = MultiplexSDA.features_dataframe.copy()

            mp_params_for_sda_prob.append({
                "sda_prob_params": (net_size, char_dist, homophily),
                "networks_dataframe": networks_dataframe,
                "features_dataframe": features_dataframe
            })

            for net1_deg_seq_param in net1_deg_seq_params:
                for net2_deg_seq_param in net2_deg_seq_params:
                    for deg_seq_corr in deg_seq_corr_params:

                        m1, v1 = net1_deg_seq_param
                        m2, v2 = net2_deg_seq_param
                        
                        unique_id += 1
                        metadata.append({
                            "unique_id": unique_id,
                            "net1_name": network_names[0],
                            "net2_name": network_names[1],
                            "feat_name": feature_names[0],
                            "net_size": net_size,
                            "char_dist": char_dist,
                            "homophily": homophily,
                            "net1_deg_seq_mean": m1,
                            "net1_deg_seq_var": v1,
                            "net2_deg_seq_mean": m2,
                            "net2_deg_seq_var": v2,
                            "deg_seq_corr": deg_seq_corr
                        })
                        
                        for sample_id in range(number_of_samples):
                            mp_params_for_deg_seqs.append({
                                "unique_id": unique_id,
                                "sample_id": sample_id,
                                "network_names": network_names,
                                "deg_seqs_params": (net_size, m1, v1, m2, v2, deg_seq_corr)
                            })

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %% [markdown]
# **Multiprocessing: SDA Connection Probabilities**

# %%
print("⏳ [Stage] Multiprocessing SDA probabilities...", flush=True)
t = time.time()

# %%
with mp.Pool(processes=number_of_mp_cpus) as pool:
    results = pool.map(mp_worker_for_sda_prob, mp_params_for_sda_prob)

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %%
print("⏳ [Stage] Consolidating SDA probabilities...", flush=True)
t = time.time()

# %%
mp_results_for_sda_prob = {result["sda_prob_params"]: result["probabilities"] for result in results}

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %% [markdown]
# **Multiprocessing: Degree Sequences**

# %%
print("⏳ [Stage] Multiprocessing degree sequences...", flush=True)
t = time.time()

# %%
with mp.Pool(processes=number_of_mp_cpus) as pool:
    results = pool.map(mp_worker_for_deg_seqs, mp_params_for_deg_seqs)

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %%
print("⏳ [Stage] Consolidating degree sequences...", flush=True)
t = time.time()

# %%
mp_results_for_deg_seqs = {(result["unique_id"], result["sample_id"]): result["degree_sequences"] for result in results}

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %% [markdown]
# **Multiprocessing: Networks**

# %%
print("⏳ [Stage] Generating parameters for multiprocessing networks...", flush=True)
t = time.time()

# %%
for meta in metadata:

    unique_id = meta["unique_id"]
    net_size, char_dist, homophily = meta["net_size"], meta["char_dist"], meta["homophily"]
        
    probabilities = mp_results_for_sda_prob[(net_size, char_dist, homophily)]

    for sample_id in range(number_of_samples):
        
        degree_sequences = mp_results_for_deg_seqs[(unique_id, sample_id)]
    
        mp_params_for_networks.append({
            "unique_id": unique_id,
            "sample_id": sample_id,
            "network_names": network_names,
            "probabilities": probabilities,
            "degree_sequences": degree_sequences,
            "compare_models": compare_models,
            "compute_metrics": compute_metrics
        })

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %%
print("⏳ [Stage] Multiprocessing networks...", flush=True)
t = time.time()

# %%
start_time = time.time()

with mp.Pool(processes=number_of_mp_cpus) as pool:
    results = pool.map(mp_worker_for_networks, mp_params_for_networks)

end_time = time.time()
duration = end_time - start_time

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %% [markdown]
# **Complete: Convert to DataFrames and Save as Parquet Files**

# %%
print("⏳ [Stage] Converting to DataFrames...", flush=True)
t = time.time()

# %%
results = [dictionary for sample in results for dictionary in sample]
results = pd.DataFrame(results)
results = results.sort_values(by=["unique_id", "sample_id", "model_id"])
results = results.reset_index(drop=True)

metadata = pd.DataFrame(metadata)
metrics = results.drop(columns=[col for col in results.columns if "distributions" in col])
distributions = src.data.joint_distributions(dataframe=results)

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)

# %%
filepath = "../../results/synthetic"
filename = f"{time.strftime('%Y-%m-%d', time.localtime())}_synthetic_duration-{duration:.0f}_samples-{number_of_samples}"

# %%
print("⏳ [Stage] Saving files...", flush=True)
t = time.time()

# %%
metadata.to_parquet(f"{filepath}/{filename}_metadata.parquet", compression="snappy")
metrics.to_parquet(f"{filepath}/{filename}_metrics.parquet", compression="snappy")
distributions.to_parquet(f"{filepath}/{filename}_distributions.parquet", compression="snappy")

locations = {f"net_size_{len(loc)}": loc for loc in locations}
np.savez(f"{filepath}/{filename}_locations.npz", **locations)

# %%
print(f"✅ Completed in {time.time() - t:.2f} seconds.\n", flush=True)


