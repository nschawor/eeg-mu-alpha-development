"""Use bycycle to compute features of cycles in each component's time series.
"""
# Import necessary modules
from bycycle.features import compute_features
import time
import pandas as pd
import multiprocessing as mp
from core.helper import load_comp_timeseries, get_subject_list, \
    get_alpha_mu_comps
from core.params import *


def compute_features_one_comp(
        i, s, fs, freq_band, burst_kwargs, verbose=BYCYCLE_VERBOSE):
    """Compute bycycle features for one component.

    Parameters
    ----------
    i : int
        Component index.
    s : array
        Component time series.
    fs : float
        Sampling frequency.
    freq_band : tuple of float
        Frequency band to compute bycycle features.
    burst_kwargs : dict
        Burst detection parameters.
    verbose : bool
        Whether to print out progress.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with bycycle features for each cycle in the component.
    """
    start = time.time()
    df = compute_features(
        s, fs, freq_band, threshold_kwargs=burst_kwargs)
    df['comp'] = i
    if verbose:
        print('Processed Component #{} with shape {} in {} seconds'.format(
            i, s.shape, time.time() - start))
    return df


def compute_bycycle_features(
        comps, df, fs, peak_range=PEAK_RANGE, n_processes=N_PROCESSES,
        burst_kwargs=BURST_KWARGS):
    """Compute bycycle features for each given component.

    Parameters
    ----------
    comps : array
        Array of component time series.
    df : pd.DataFrame
        DataFrame with component indices and peak frequencies.
    fs : float
        Sampling frequency.
    peak_range : float
        Range around peak frequency to compute bycycle features.
    n_processes : int
        Number of processes to use for multiprocessing.
    burst_kwargs : dict
        Burst detection parameters.

    Returns
    -------
    df_cycles : pd.DataFrame
        DataFrame with bycycle features for each cycle in each component.
    """
    # Compute bycycle features for each channel using multiprocessing
    f_bands = [(int(i), (pf - peak_range / 2, pf + peak_range / 2))
               for _, (i, pf) in df[['i_comp', 'PF']].dropna().iterrows()]
    args = [(i, comps[i, :], fs, f_band, burst_kwargs) for i, f_band in
            f_bands]

    with mp.Pool(processes=n_processes) as pool:
        dfs = pool.starmap(compute_features_one_comp, args)

    if not dfs:
        return

    df_cycles = pd.concat(dfs)
    return df_cycles


def process_one_subj(
        subj, comps_idx, bycycle_folder=BYCYCLE_FOLDER,
        ssd_sparam_folder=SSD_SPARAM_FOLDER, verbose=BYCYCLE_VERBOSE):
    """Compute waveform features for one subject using bycycle package.

    Parameters
    ----------
    subj : str
        Subject ID.
    comps_idx : list
        List of component indices to process.
    bycycle_folder : str
        Path to folder to save bycycle output.
    ssd_sparam_folder : str
        Path to folder with SSD spectral parameters.
    verbose : bool
        Whether to print out progress.
    """
    # Start timer
    start = time.time()

    # Determine filenames for SSD, raw, and figure to be saved
    ssd_param_fn = f'{ssd_sparam_folder}/{subj}_spec_param.csv'

    # Get components' peak frequency
    comps_df = pd.read_csv(ssd_param_fn)

    # Select components to process
    comps_df = comps_df.loc[comps_df['i_comp'].isin(comps_idx)]

    # Load component time series
    comps, fs = load_comp_timeseries(subj)

    # Compute bycycle features
    bycycle_df = compute_bycycle_features(comps, comps_df, fs)
    if bycycle_df is None:
        return

    # Save bycycle DataFrame as csv
    bycycle_df.to_csv(f'{bycycle_folder}/{subj}.csv')

    if verbose:
        print("It took {} secs to process {}'s data".format(
            time.time() - start, subj))


def process_all_subjs(bycycle_folder=BYCYCLE_FOLDER, verbose=BYCYCLE_VERBOSE):
    """Compute waveform features for all subjects using bycycle package.

    Parameters
    ----------
    bycycle_folder : str
        Path to folder to save bycycle output.
    verbose : bool
        Whether to print out progress.
    """
    # Get subjects
    subjs = get_subject_list('sources')

    # Make directory to save bycycle output (if necessary)
    if not os.path.exists(bycycle_folder):
        os.makedirs(bycycle_folder)

    # Report number of files that will be processed in the end, the number
    # already processed, and how many still need to be processed
    n_total, n_processed = len(subjs), len(set(
        [f for f in os.listdir(bycycle_folder) if not f.startswith('.')]))
    if verbose:
        print('\nFILE COUNT\nMeet Amplitude Cutoff: {}'
              '\nAlready Processed: {}\nStill to Process: {}\n'.format(
                n_total, n_processed, n_total - n_processed))

    # Remove already processed from list to be processed
    all_comps_df = pd.concat(get_alpha_mu_comps())
    subjs_to_process = [subj for subj in subjs if not os.path.exists(
        f'{bycycle_folder}/{subj}.csv')]

    for i, subj in enumerate(subjs_to_process):
        comps_idx = all_comps_df.query('subject == @subj')['idx'].to_list()
        process_one_subj(subj, comps_idx, verbose=verbose)
        if verbose:
            print('\nProcessed {} of {} Files!!\n'.format(
                i + n_processed + 1, n_total))


if __name__ == '__main__':
    process_all_subjs()
