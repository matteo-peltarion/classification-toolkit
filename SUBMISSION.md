# Submission

Here are a few additional (probably useless) info about my submission.

## Report

The report is available as a jupyter notebook (file `report.ipynb` in folder
`notebooks`) and as an exported self-contained html page (in case of emergency)
in the same `notebooks` folder (file `report.html`).

## Replicating experiments

In case you want to try the code yourself, here are a few instructions on how to
do it.

### Setup

#### Conda environment

Setup the conda environment with the following command (conda must be already
installed):

```shell
conda env create -f environment.yml
```

#### Dataset download

Use script `download_data.sh` to download and unpack in the right folder the
dataset used in this assignment.

```shell
download_data.sh
```

### Launching experiments

You can either launch file `main.py` directly (possibly passing the -h argument
to see all available options) or copy the included bash script
`train_command.sh` and edit it to change the experiment's configuration, and
then launch that directly with

```shell
./your-copy-of-train_command.sh
```

The appropriate conda environment `lesions-classification` must have been
activated before launching the experiments.
