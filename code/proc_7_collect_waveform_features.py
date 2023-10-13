"""Collect waveform features for all components across all subjects into one
big DataFrame for further analysis and plotting.
"""
# Import necessary modules
import pandas as pd
import numpy as np
from core.helper import get_alpha_mu_comps, filter_by_thresholds
from core.params import *


def extract_waveform_features_one_rhythm(
        comps_df, bycycle_feats=BYCYCLE_FEATS_TO_EXTRACT,
        bycycle_folder=BYCYCLE_FOLDER, ssd_sparam_folder=SSD_SPARAM_FOLDER,
        s_freq=S_FREQ):
    """Extract waveform features for one rhythm type (alpha or mu) across all
    subjects.

    Parameters
    ----------
    comps_df : pd.DataFrame
        DataFrame with component indices and peak frequencies.
    bycycle_feats : list of str
        List of bycycle features to extract.
    bycycle_folder : str
        Path to folder with bycycle output.
    ssd_sparam_folder : str
        Path to folder with SSD spectral parameters.
    s_freq : float
        Sampling frequency.

    Returns
    -------
    waveform_feats_df_one_rhythm : pd.DataFrame
        DataFrame with waveform features for one rhythm type across all
        subjects.
    """
    # Initialize DataFrame for extracted features
    waveform_feats_df_one_rhythm = pd.DataFrame([])
    for subj_id, i_comp, snr, pattern_dist in comps_df.itertuples(
            index=False):
        # Load bycycle DataFrame
        bycycle_f = f'{bycycle_folder}/{subj_id}.csv'
        bycycle_df = pd.read_csv(bycycle_f)

        # Subselect desired bycycle features for component
        bycycle_df = bycycle_df[bycycle_df['comp'] == i_comp]
        bycycle_df = bycycle_df[bycycle_feats + ['is_burst']]

        # Skip if no bursts
        if len(bycycle_df) == 0:
            continue

        # Transform symmetry measures to asymmetry measures
        sym_feats = [feat for feat in bycycle_df.columns if 'sym' in feat]
        for feat in sym_feats:
            bycycle_df[feat.replace('sym', 'asym_cyc')] = np.abs(
                bycycle_df[feat] - 0.5)

        # Calculate frequency from period
        if 'period' in bycycle_df.columns:
            bycycle_df['frequency'] = s_freq / bycycle_df['period']

        # Determine average amplitude of non-bursts
        bycycle_df['non_burst_amp_mean'] = bycycle_df['volt_amp'].where(
            ~bycycle_df['is_burst']).mean()

        # Subselect bursts
        bycycle_df = bycycle_df.query('is_burst').reset_index(drop=True)

        # Count number of cycyles
        bycycle_df['n_cycles'] = len(bycycle_df)

        # Take mean of features across bursts
        bycycle_df = bycycle_df.mean().to_frame().T

        # Calculate asymmetry from mean symmetry, not as the mean of cycle
        # asymmetry measures
        sym_feats = [feat for feat in bycycle_df.columns if 'sym' in
                    feat and 'asym' not in feat]
        for feat in sym_feats:
            bycycle_df[feat.replace('sym', 'asym')] = np.abs(
                bycycle_df[feat] - 0.5)

        # Load component SNR, exponent, and offset
        ssd_sparam_f = f'{ssd_sparam_folder}/{subj_id}_spec_param.csv'
        ssd_sparam_df = pd.read_csv(ssd_sparam_f)
        comp_exp = ssd_sparam_df[ssd_sparam_df['i_comp'] == i_comp][
            'exponent'].copy().values[0]
        comp_offset = ssd_sparam_df[ssd_sparam_df['i_comp'] == i_comp][
            'offset'].copy().values[0]

        # Add data to big DataFrame
        bycycle_df = pd.concat(
            [pd.DataFrame({'EID': [subj_id], 'comp': [i_comp], 'SNR': [snr],
             'pattern_dist': [pattern_dist], 'comp_exp': [comp_exp],
             'comp_offset': [comp_offset]}), bycycle_df], axis=1)
        waveform_feats_df = bycycle_df.loc[
            :, ~bycycle_df.columns.duplicated()].copy()
        waveform_feats_df_one_rhythm = waveform_feats_df_one_rhythm.append(
            waveform_feats_df, ignore_index=True)
    return waveform_feats_df_one_rhythm


def get_waveform_features_all_comps(
        all_subjects_csv=ALL_SUBJECTS_CSV,
        save_fname=WAVEFORM_FEATS_CSV, snr_threshold=SNR_THRESHOLD,
        pattern_dist_threshold=PATTERN_DIST_THRESHOLD):
    """Collect waveform features for all components across all subjects into one
    big DataFrame for further analysis and plotting.

    Parameters
    ----------
    all_subjects_csv : str
        Path to CSV with all subjects.
    save_fname : str
        Path to save waveform features DataFrame.
    snr_threshold : float
        SNR threshold for filtering components.
    pattern_dist_threshold : float
        Pattern distance threshold for filtering components.

    Returns
    -------
    all_waveform_feats : pd.DataFrame
        DataFrame with waveform features for all components across all subjects.
    """
    # Get alpha and mu components
    alpha_comps, mu_comps = get_alpha_mu_comps()

    # Extract bycycle features for alpha and mu components
    alpha_waveform_feats = extract_waveform_features_one_rhythm(alpha_comps)
    mu_waveform_feats = extract_waveform_features_one_rhythm(mu_comps)

    # Load ages from CSV
    ages = pd.read_csv(all_subjects_csv, index_col='EID')['Age']

    # Combine age and peak trough symmetry into one DataFrame
    alpha_waveform_feats['Age'] = alpha_waveform_feats['EID'].map(ages)
    mu_waveform_feats['Age'] = mu_waveform_feats['EID'].map(ages)

    # Combine alpha and mu commponents into one DataFrame
    alpha_waveform_feats['Rhythm'] = 'Alpha'
    mu_waveform_feats['Rhythm'] = 'Mu'
    all_waveform_feats = pd.concat(
        [alpha_waveform_feats, mu_waveform_feats], ignore_index=False)

    # Normalize amplitude within each subject
    if 'volt_amp' in all_waveform_feats.columns:
        all_waveform_feats['amp_norm'] = all_waveform_feats[
            'volt_amp'] / all_waveform_feats['non_burst_amp_mean']

    # Save bycycle features DataFrame to CSV
    all_waveform_feats.to_csv(save_fname, index=False)

    # Filter by SNR and pattern distance thresholds
    all_waveform_feats = filter_by_thresholds(
        all_waveform_feats, snr_threshold=snr_threshold,
        pattern_dist_threshold=pattern_dist_threshold)

    # Take top component for each rhythm type
    all_waveform_feats = all_waveform_feats.loc[all_waveform_feats.groupby(
        ['EID', 'Rhythm'])['comp'].idxmin()].reset_index(drop=True)
    return all_waveform_feats


if __name__ == '__main__':
    # Plot bycycle features
    waveform_feats = get_waveform_features_all_comps()