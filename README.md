# Resting-state is not enough: alpha and mu rhythms change shape across development, but lack diagnostic sensitivity

This repository contains all the necessary code to reproduce the analysis and figures of the following manuscript:

- Bender, Voytek, & Schaworonkow: Resting-state is not enough: alpha and mu rhythms change shape across development, but lack diagnostic sensitivity]. *bioRxiv* (2023). doi: https://doi.org/10.1101/2023.10.13.562301.

## Dataset

The results are based on an [openly available developmental dataset](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/index.html) of resting-state electroencephalography (EEG) recordings from over 2500 children (ages 5-18) from the [Child Mind Institute](https://childmind.org/) published by Alexander and colleagues (2017):

**Alexander et al.: [An open resource for transdiagnostic research in pediatric mental health and learning disorders](https://www.nature.com/articles/sdata2017181) *Scientific Data* (2017). doi: 10.1038/sdata.2017.181**.

The dataset is a part of the [Healthy Brain Network](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/About.html), an ongoing initiative focused on creating a transdiagnostic biobank of data from over 10,000 children and adolescents (ages 5-21). The whole dataset contains EEG, magnetic reasonance imaging (MRI), and phenotypic data, any of which can be downloaded [here](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/sharing_neuro.html). Some participants did not complete all assessments, so the number of participants varies depending on the particular type of data one wishes to analyze.

To reproduce our results for Figures 1-3, the EEG data for each of the 10 releases should be downloaded and placed in the `data/download` folder. These data are distributed under the [Creative Commons, Attribution Non-Commercial Share Alike License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

To reproduce our results for Figure 4, the phenotypic data containing participant diagnoses must be downloaded. Details on how to access this phenotypic data are provided [here](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Pheno_Access.html). The diagnostic information should be contained in a CSV file, consisting of `subject` and `diagnosis` columns, with each row containing the subject ID and diagnosis for one participant. The path to this CSV file should be set as the `DIAGNOSES_CSV` variable in the `code/params.py` file.
## Requirements

The provided Python 3 scripts require the following packages:
- `numpy` and `scipy` for numerical computation
- `pandas` for storage and manipulation of tabular data
- `mne` for reading and storing EEG data
- [`fooof`](https://fooof-tools.github.io/fooof/) for parameterizing neural power spectra
- [`bycycle`](https://bycycle-tools.github.io/bycycle/) for computing waveform shape features from neural time series
- `matplotlib`, `pyvista`, `scikit_image`, and `seaborn` for visualizing data and generating figures
- `pinguoin` and `statannotations` for calculating and visualizing statistics

The particular versions used to perform the analysis and create the figures for the manuscript are specified in the `requirements.txt` file. To install all these packages in one's local Python environment, simplify enter `pip install -r requirements.txt` into the command line from the root of this GitHub repository.

## Processing Pipeline

Alpha and mu rhythms are mixed in sensor-space EEG due to volume conduction. In our analysis, we isolated these alpha-band rhythms from resting-state EEG using spatio-spectral decomposition ([Nikulin et al., 2011](https://pubmed.ncbi.nlm.nih.gov/21276858/)) and template matching:![](./figures/fig1_methods.png)

Our processing pipeline can be run from start to finish by running the `code/proc_*.py` files in order. The output files `sources.csv` and `waveform_feats.csv` are provided in the `results` folder if one wishes to avoid downloading the data and processing the entire pipeline. All parameters used to generate these output files are contained in the `core/params.py` file.

To see how the results and figures change with different choices of parameters, change the parameters in the `core/params.py` file and rerun the necessary `code/proc_*.py` files.

## Figures

To generate figures from the manuscript, simply run the corresponding `code/fig*.py` file(s). Figures will be saved in the `figures` folder.
