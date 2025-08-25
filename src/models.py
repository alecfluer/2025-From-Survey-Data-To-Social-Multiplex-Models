import pathlib
import sys

import networkx as nx
import numpy as np
import sdcdp

sys.path.insert(0, f"{str(pathlib.Path(__file__).resolve().parent.parent)}/external/sda-model-master")
import sdnet

def configuration_undirected(degree_sequence, probabilities):
    return nx.configuration_model(
        deg_sequence=degree_sequence,
        create_using=nx.MultiGraph()
    )

def configuration_directed(degree_sequence, probabilities):
    return nx.directed_configuration_model(
        in_degree_sequence=degree_sequence[1],
        out_degree_sequence=degree_sequence[0],
        create_using=nx.MultiDiGraph()
    )

def sdcdp_undirected(degree_sequence, probabilities):
    return sdcdp.sdcdp.sdcdp_model(degrees=degree_sequence, probabilities=probabilities, directed=False, simple=False)

def sdcdp_directed(degree_sequence, probabilities):
    return sdcdp.sdcdp.sdcdp_model(degrees=degree_sequence, probabilities=probabilities, directed=True, simple=False)

def sdc_undirected(degree_sequence, probabilities):
    model = sdnet.SDA(P=probabilities, k=None, b=None, alpha=None, p_rewire=0, directed=False)
    model.set_degseq(degseq=degree_sequence, sort=False)
    model = model.conf_model(simplify=False, sparse=False)
    model[np.diag_indices_from(model)] //= 2
    return nx.from_numpy_array(model, parallel_edges=True, create_using=nx.MultiGraph())

def sdc_directed(degree_sequence, probabilities):
    model = sdnet.SDA(P=probabilities, k=None, b=None, alpha=None, p_rewire=0, directed=True)
    model.set_degseq(degseq=degree_sequence.T, sort=False)
    model = model.conf_model(simplify=False, sparse=False)
    return nx.from_numpy_array(model, parallel_edges=True, create_using=nx.MultiDiGraph())