"""Determine alpha peak frequency from raw continuous recordings.
"""
# %% Imports
import mne
import fooof
from fooof.core.errors import DataError, NoModelError
from fooof.analysis import get_band_peak_fg
import pandas as pd
from core.params import *
from core.helper import apply_laplacian, get_subject_list, get_SNR


def sparam_one_subj(
        subject, mne_raw_folder=MNE_RAW_FOLDER,
        laplacian_channels=LAPLACIAN_CHANNELS, fit_fmin=FIT_FMIN,
        fit_fmax=FIT_FMAX, peak_fmin=PEAK_FMIN, peak_fmax=PEAK_FMAX,
        n_peaks=N_PEAKS, peak_width_limits=PEAK_WIDTH_LIMITS):
    """Perform spectral parameterization on specified channels for one subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    mne_raw_folder : str
        Folder where MNE raw files are saved.
    laplacian_channels : dict
        Dictionary with keys as channel names and values as lists of channel
        names to use for Laplacian.
    fit_fmin : float
        Minimum frequency to fit FOOOF model.
    fit_fmax : float
        Maximum frequency to fit FOOOF model.
    peak_fmin : float
        Minimum frequency to extract peak frequency from FOOOF model.
    peak_fmax : float
        Maximum frequency to extract peak frequency from FOOOF model.
    n_peaks : int
        Number of peaks to fit with FOOOF.
    peak_width_limits : tuple of float
        Limits on peak width to fit with FOOOF.
    """
    os.makedirs(SENSOR_SPARAM_FOLDER, exist_ok=True)
    df_fname = f'{SENSOR_SPARAM_FOLDER}/{subject}_spec_param.csv'

    # create empty dataframe
    peaks = np.zeros((1, 2 * (len(laplacian_channels) + 1)))
    columns = ["subject", "peak", "power"]
    columns.extend(["%s %s" % (chan, desc) for chan in laplacian_channels
                    for desc in ["peak", "power"]])
    df = pd.DataFrame(
        np.hstack(([[subject]], peaks)), columns=columns)
    df = df.set_index('subject')

    # Determine file name for raw
    raw_fname = f'{mne_raw_folder}/{subject}-raw.fif'

    # Load in raw file, put in NaNs if unable to
    try:
        raw = mne.io.read_raw_fif(raw_fname, preload=True)
    except (ValueError, AttributeError):
        df.loc[subject] = subject, *[np.nan] * 2 * (len(
            laplacian_channels) + 1)
        df['subject'] = df.index
        df.to_csv(df_fname, index=False)
        print('Raw sucks')

    # Apply laplacian
    raw.pick_types(eeg=True)
    processed_raw = apply_laplacian(raw, laplacian_channels)

    # Calculate PSD and fit FOOOF, put in NaNs if unable to
    n_fft = int(NR_SECONDS_SPEC*raw.info['sfreq'])
    psd_all, freqs = mne.time_frequency.psd_welch(
        processed_raw, n_fft=n_fft, fmin=fit_fmin, fmax=fit_fmax)
    fm = fooof.FOOOFGroup(
        max_n_peaks=n_peaks, peak_width_limits=peak_width_limits)
    try:
        fm.fit(freqs, psd_all)
    except (DataError, NoModelError):
        df.loc[subject] = [np.nan] * 2 * (len(laplacian_channels) + 1)
        df['subject'] = df.index
        df.to_csv(df_fname, index=False, na_rep='nan')
        print('FOOOF sucks')

    # Extract peak frequency and power from FOOOF
    peak_params = get_band_peak_fg(fm, [peak_fmin, peak_fmax])
    peak_frequency = np.nanmean(peak_params[:, 0])
    peak_power = get_SNR(
        processed_raw, fmin=fit_fmin, fmax=fit_fmax,
        freq=[peak_fmin, peak_fmax], seconds=NR_SECONDS_SPEC)
    df.loc[subject] = peak_frequency, np.nanmean(
        peak_power), *peak_params[:, :2].flatten()

    # save subject specific series
    df['subject'] = subject
    df.to_csv(df_fname, index=False)


def sparam_all_subjs(
        sensor_sparam_folder=SENSOR_SPARAM_FOLDER,
        results_folder=RESULTS_FOLDER,
        laplacian_channels=LAPLACIAN_CHANNELS):
    """Perform spectral parameterization on specified channels for all subjects.

    Parameters
    ----------
    sensor_sparam_folder : str
        Folder where spectral parameterization results are saved.
    results_folder : str
        Folder where results are saved.
    laplacian_channels : dict
        Dictionary with keys as channel names and values as lists of channel
        names to use for Laplacian.
    """
    # Make results directory if necessary
    os.makedirs(results_folder, exist_ok=True)

    # Get all subjects for which there is a resting state file
    subjects = get_subject_list('file')

    # Perform spectral parameterization for each subject
    dfs = []
    for i_sub, subject in enumerate(subjects):
        print(f"{i_sub}/{len(subjects)}: {subject}")

        df_fname = f'{sensor_sparam_folder}/{subject}_spec_param.csv'
        if os.path.exists(df_fname):
            continue
        sparam_one_subj(subject, laplacian_channels=laplacian_channels)

    # Collect all files
    dfs = []
    for i_sub, subject in enumerate(subjects):
        print(f"{i_sub:04}/{len(subjects)}: {subject}")

        df_fname = f'{sensor_sparam_folder}/{subject}_spec_param.csv'
        if os.path.exists(df_fname):
            df = pd.read_csv(df_fname)
            dfs.append(df)

    # Save out peak frequencies for all subjects
    df_all = pd.concat(dfs)
    print(f'nr_subjects = {len(df_all)}')
    df_all.to_csv(
        f'{RESULTS_FOLDER}/subjects_with_peak_frequency.csv', index=False)


if __name__ == '__main__':
    sparam_all_subjs()
# %%
