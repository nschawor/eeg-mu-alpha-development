"""Determine spectral parameters for SSD components.
"""
# %% Imports
import mne
import pandas as pd
import fooof
from fooof.analysis import get_band_peak_fg
import core.ssd
from core.params import *
from core.helper import get_SNR, get_subject_list


def sparam_on_ssd_components(
        ssd_folder=SSD_FOLDER, ssd_sparam_folder=SSD_SPARAM_FOLDER,
        mne_raw_folder=MNE_RAW_FOLDER, num_components=NUM_COMPONENTS,
        peak_fmin=PEAK_FMIN, peak_fmax=PEAK_FMAX, fit_fmin=FIT_FMIN,
        fit_fmax=FIT_FMAX, n_peaks=N_PEAKS,
        peak_width_limits=PEAK_WIDTH_LIMITS):
    """Determine spectral parameters for SSD components.

    Parameters
    ----------
    ssd_folder : str
        Path to folder with SSD filters and patterns.
    ssd_sparam_folder : str
        Path to folder to save SSD spectral parameters.
    mne_raw_folder : str
        Path to folder with MNE raw files.
    num_components : int
        Number of SSD components to use.
    peak_fmin : float
        Minimum frequency to extract peak frequency from FOOOF model.
    peak_fmax : float
        Maximum frequency to extract peak frequency from FOOOF model.
    fit_fmin : float
        Minimum frequency to fit FOOOF model.
    fit_fmax : float
        Maximum frequency to fit FOOOF model.
    n_peaks : int
        Number of peaks to fit with FOOOF.
    peak_width_limits : tuple of float
        Limits on peak width to fit with FOOOF.
    """
    # Make folder to save SSD specparam files
    os.makedirs(ssd_sparam_folder, exist_ok=True)
    subjects = get_subject_list('ssd')
    for i_sub, subject in enumerate(subjects):
        # Print subject upon the start of processing
        print('%s (%d/%d)' % (subject, i_sub + 1, len(subjects)))

        # Determine file names for necessary files
        ssd_filters_fname = f'{ssd_folder}/{subject}_ssd_filters.csv'
        ssd_patterns_fname = ssd_filters_fname.replace('filters', 'patterns')
        df_fname = f'{ssd_sparam_folder}/{subject}_spec_param.csv'
        raw_fname = f'{mne_raw_folder}/{subject}-raw.fif'

        # Skip if processing already done
        if os.path.exists(df_fname):
            continue

        # Load the raw file
        raw = mne.io.read_raw_fif(raw_fname)
        raw.load_data()
        raw.pick_types(eeg=True)

        # Load SSD filters and patterns
        filters = pd.read_csv(ssd_filters_fname).values.T[:, :num_components]
        patterns = pd.read_csv(ssd_patterns_fname).values.T[:, :num_components]

        # Apply filters
        raw_ssd = core.ssd.apply_filters(raw, filters)

        # Compute psd and threshold criterion
        n_fft = int(NR_SECONDS_SPEC * raw.info['sfreq'])
        psd, freq = mne.time_frequency.psd_welch(
            raw_ssd, fmin=1, fmax=45, n_fft=n_fft)

        # Fit FOOOF
        fm = fooof.FOOOFGroup(
            max_n_peaks=n_peaks, peak_width_limits=peak_width_limits)
        fm.fit(freq, psd)

        # Take above the threshold
        peaks = get_band_peak_fg(fm, [peak_fmin, peak_fmax])
        subject_ids = [subject] * peaks.shape[0]
        which_pattern = np.arange(peaks.shape[0])

        # Extract offset and aperiodic for component
        exponent = fm.get_params('aperiodic_params', 'exponent')
        offset = fm.get_params('aperiodic_params', 'offset')

        # Save out SpecParam and SSD patterns
        peak_power = get_SNR(
            raw_ssd, fmin=fit_fmin, fmax=fit_fmax,
            freq=[peak_fmin, peak_fmax], seconds=NR_SECONDS_SPEC)
        data = np.array((
            subject_ids, which_pattern, peaks[:, 0], peak_power, exponent,
            offset))
        columns = ["subject_id", "i_comp", "PF", "SNR", "exponent", "offset"]
        df_selected = pd.DataFrame(data.T, columns=columns)
        if len(df_selected.query("PF != 'nan'")) == 0:
            print(f"No peak found for {subject}'s SSD components")
            continue
        df_selected.to_csv(df_fname, index=False)
        df_patterns = pd.DataFrame(patterns.T, columns=raw.ch_names)
        df_patterns.to_csv(
            df_fname.replace('_spec_param', '_patterns'), index=False)


if __name__ == '__main__':
    sparam_on_ssd_components()
# %%
