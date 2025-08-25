# **From Survey Data to Social Multiplex Models**

## **Overview**

This repository accompanies the manuscript [*From survey data to social multiplex models: Incorporating interlayer correlation from multiple data sources*](#references). It includes the results and figures presented in the manuscript. It provides simulation and analysis workflows for generating and evaluating social multiplex networks from both synthetic data and survey-derived data.

## **Authors**

- Alec Fluer, Ian Laga, Logan Graham, Breschine Cummins  
    Department of Mathematical Sciences, Montana State University, Bozeman, MT, USA  
    - Lead author: alecfluer@montana.edu  
    - Corresponding author: breschine.cummins@montana.edu

- Ellen Almirol  
    University of Chicago, Chicago, IL, USA

- Makenna Meyer  
    North Carolina State University, Raleigh, NC, USA

## **Data**

Data includes reconstructed degree sequences from uConnect survey responses and merged U.S. Census Bureau data. Pending DUA.

## **Results**

The results and figures presented in the manuscript for the uConnect survey data are provided. This is to provide an example of the output files and allow for ready inspection.

## **Reproducibility**

There are identical workflows for the synthetic data and uConnect survey data. A placeholder is marked with an asterisk (`*`).

To generate simulation results, run `simulate_*.ipynb` or `simulate_*.py`. These are identical and will result in four output files: a metadata file, metrics file, distributions file, and locations file. The metadata file contains information about the simulation run. The metrics file contains computed global network metrics. The distributions file contains computed local network metrics. The locations file contains sampled node locations. The files are associated with a common filename identifier. The common filename identifier includes the date, type of data (`*`), approximate duration of the simulation run, and number of samples in the simulation run.

To analyze simulation results and generate figures, run `analyze_*.ipynb`. This will require specifying the common filename identifier.

The `models.py` and `workers.py` modules, and the `simulate_*.ipynb` or `simulate_*.py` script, were developed to be customizable. They can be modified to adjust sampling logic and metric computation.

## **Scripts**

- `simulate_synthetic.ipynb`: Runs simulations on synthetic data using multiprocessing. Results are saved in `results/synthetic/`.
- `analyze_synthetic.ipynb`: Generates figures using the synthetic data simulation results. Figures are saved in `results/synthetic/plots/`.
- `preprocess_uconnect.ipynb`: Converts the uConnect survey data and U.S. Census Bureau data into usable formats.
- `simulate_uconnect.ipynb`: Runs simulations on the uConnect survey data using multiprocessing. Results are saved in `results/uconnect/`.
- `analyze_uconnect.ipynb`: Generates figures using the uConnect survey data simulation results. Figures are saved in `results/uconnect/plots/`.

## **Modules**

- `data.py`: Provides functions for handling data and computing metrics post-simulation.
- `models.py`: Provides functions for sampling networks.
- `plots.py`: Provides functions for generating figures.
- `workers.py`: Provides functions for multiprocessing.

## **Dependencies**

Requires Python ≥ 3.8 and the following packages:

- `fastparquet`
- `geopandas`
- `matplotlib`
- `mpl_toolkits`
- `networkx`
- `numpy`
- `pandas`
- `pyarrow`
- `scipy`
- `seaborn`
- `shapely`

In addition, the following (which may have their own dependencies) are required:

- `sdcdp`: A custom package available at [https://github.com/alecfluer/sdcdp](https://github.com/alecfluer/sdcdp)
- `sdnet`: An external model included in `external/sda-model-master/`

## **Runtime**

Computational efforts were performed on the Tempest High Performance Computing System, operated and supported by University Information Technology Research Cyberinfrastructure (RRID:SCR_026229) at Montana State University. Simulations were run with 200 CPUs and 750 MB of memory, with runtime up to 6 hours.

## **License**

Copyright © 2025 Alec Fluer  
This software is licensed under the MIT License. See LICENSE for details.

## **References**

Alec Fluer, Ian Laga, Logan Graham, Ellen Almirol, Makenna Meyer, and Breschine Cummins.  
From survey data to social multiplex models: Incorporating interlayer correlation from multiple data sources. In preparation.