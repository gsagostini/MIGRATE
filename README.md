# MIGRATE

This code accompanies the paper `Inferring Fine-Grained Migration Patterns Across the United States` ([current version](https://arxiv.org/abs/2503.20989)). If using either the data or the code, please cite the paper:

```
Agostini, G., Young, R., Fitzpatrick, M., Garg, N., & Emma, P. (2025). Inferring Fine-Grained Migration Patterns Across the United States. https://doi.org/10.48550/arXiv.2503.20989
```

## Data availability

If you would like to request access to the MIGRATE dataset, follow the instructions on the [project website](https://migrate.tech.cornell.edu). We grant data access to academic and non-profit research use.

Some datasets are missing from this code release because they are proprietary (e.g. raw Infutor data will not be provided) or heavy yet publicly available (e.g. raw Census geographies). The paper contains citations to datasets, and some notebooks in the repo note where to obtain the raw data. Feel free to email gsagostini@infosci.cornell.edu with any questions.

## Installation

A `.yml` file is provided with all required packages and their versions. Please update the `d03_src.vars.py` file with your path to the repository `_path_to_repo` to ensure all functions work correctly.

## Repository Setup

```
│
├── d01_data                                      <- Ommitted from the public release---check individual notebooks and source code for notes
│
├── d02_notebooks                                 <- Jupyter notebooks that analyse results and produce figures for the paper
│   ├── 1_Validations.ipynb                       <- Validate MIGRATE outputs (refer to section 2.2 of the paper)
│   ├── 2_National-Summaries.ipynb                <- Produce national-level migration summaries (refer to sections 2.1 and 2.3 of the paper)
│   ├── 3_Wildfires.ipynb                         <- Analyze migration in response to wildfires (refer to section 2.4 of the paper)
│   └── 4_Public-Housing.ipynb                    <- Analyze migration to and from New York City Housing Authority properties (refer to Appendix G of the paper)
│
├── d03_src                                       <- Source code for use in this project, which can be imported as modules into the notebooks and scripts
│
└─── d04_scripts                                  <- Full code routines to process address history data + fit the models
    ├── d01_read-files                            <- Scripts to read raw address histories
    ├── d02_process-addresses                     <- Scripts to geocode and clean address histories
    ├── d03_process-flows                         <- Scripts to process address histories into flow matrices
    └── d04_optimization                          <- Scripts to run our model            
```
