# Inferring Migration Flows Workshop outline

Welcome to the workshop! Here I have all the links, datasets, and notebooks for our interactive activities. If you would like to checkout the paper after the workshop, this is the most [current version](https://arxiv.org/abs/2503.20989).

## Installation

### Option 1: cloning the repository (preferred method)

Clone only this branch (otherwise you will have a lot of files!):

```git clone -b workshop --single-branch https://github.com/gsagostini/MIGRATE.git```

Create a conda environment with geopandas and supporting jupyter notebooks using the following command:

```conda create -n workshop_env python=3.13.5 geopandas numpy pandas matplotlib ipykernel jupyterlab -c conda-forge```

Then remember to activate the environment `workshop_env` before starting a notebook, for example, with VSCode or jupyter lab.

### Option 2: using Google collab

You can also choose to follow along on [Google collab](https://colab.research.google.com).

1. Go to `File > Open notebook`.
2. Select `Github`.
3. Paste the repo url: `https://github.com/gsagostini/MIGRATE/tree/workshop`.
4. Make sure you select `workshop` as the branch.
5. Open the notebook you would like to follow.

For the `mapping.ipynb` notebook, you will need to mount the drive data folder. If you follow this route, come up to me and I will share with you the drive data folder---I need your email. You need to click `Add shortcut to your drive` to add the shared `data` folder to your drive. You will then add the following cell to the top of the notebook:

```
from google.colab import drive
drive.mount('/content/drive')
```

You will log in to your drive and authenticate. Then, copy the notebook to your drive. You can then use the path `drive/MyDrive/data/` anywhere you see `data/` or `../data/`


## Activity 1: Understanding the limits of Census migration data

For this activity, you will download and follow the prompt questions to investigate the ACS county-to-county migration data portal.

1. Navigate to the ACS county-to-county [migration data portal](https://www.census.gov/topics/population/migration/guidance/county-to-county-migration-flows.html).
2. Select the 2015-2019 data file.
    <b> To think about while you look at the data: </b> what do these years represent?
3. Download the `In-, Out-, Net, and Gross Migration` file under `County-to-County`.
    - <b> Note: </b> this is an excel (.xls) file---if you don't have the software to open this file, team up with someone who does.
4. Take some time to understand the rows, columns, and sheets in the dataset.
    - <b>Discuss with your partner: </b> what column(s) (if any), would you primarily use to estimate the migration from one county to another? Do you understand what every column represents?
    - <b>Wait!</b> We will try to make sure everyone is on the same page after 5 minutes---feel free to think about the questions below, but don't bring up their answers to other groups.
    - <b>Discuss with your partner: </b> can you tell, from this data, how many people moved from Los Angeles County (CA) to New York County (NY)? To Appling County (GA)?
    - <b>Discuss with your partner: </b> what is the time period over which the migration described in the dataset is happening? As a hint, you may want to take a look at the `Inflow` file in the ACS website, and also think about the dates when the ACS was collected.

## Activity 2: Implementing Iterative Proportional Fitting --- and our variation!

For this activity, you can follow the notebook `IPF.ipynb`. Also, you may want pen and paper for the math derivations!

## Activity 3: Mapping fine-grained Migration data

For this activity, you can follow the notebook `mapping.ipynb`. You will need some publicly available datasets, which are in the `data` directory, and MIGRATE. After filling the Workshop [DUA](https://forms.gle/MCezhZGDREvbYBMK7), you received a link to the subsection of MIGRATE we will use via Google Drive. Add MIGRATE to the `data` directory if you locally cloned the repository, and . Please remember to delete the dataset afterwards.
