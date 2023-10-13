"""Compute SSD with center frequency from alpha peaks for each subject.
"""
# %% Import necessary modules
import mne
import pandas as pd
import core.ssd
from core.params import *
from core.helper import get_subject_list


def run_ssd_all_subjects(
        bin_width=SSD_BIN_WIDTH,  peak_freq_csv=PEAK_FREQ_CSV,
        ssd_folder=SSD_FOLDER, mne_raw_folder=MNE_RAW_FOLDER):
    """Compute SSD  for all subjects with individual center frequency from
    alpha peaks.

    Parameters
    ----------
    bin_width : float
        Width of bin, in Hz, to use for SSD.
    peak_freq_csv : str
        Path to CSV with peak frequencies.
    ssd_folder : str
        Path to folder to save SSD filters and patterns.
    mne_raw_folder : str
        Path to folder with MNE raw files.
    """
    # Get all subjects
    subjects = get_subject_list('sparam')

    # Make directory for SSD filters to be saved to
    os.makedirs(ssd_folder, exist_ok=True)

    for i_sub, subject in enumerate(subjects):
        # Print subject progress
        print('%s (%d/%d)' % (subject, i_sub + 1, len(subjects)))

        # Get peak frequency from CSV
        df = pd.read_csv(peak_freq_csv)
        peak = df[df.subject == subject].peak.values[0]

        # If no peak, skip subject
        if np.isnan(peak):
            continue

        # Skip subject if already processed
        ssd_filters_fname = f'{ssd_folder}/{subject}_ssd_filters.csv'
        ssd_patterns_fname = ssd_filters_fname.replace('filters', 'patterns')
        if os.path.exists(ssd_filters_fname) and os.path.exists(
                ssd_patterns_fname):
            continue

        # Load raw
        raw_fname = f'{mne_raw_folder}/{subject}-raw.fif'
        try:
            raw = mne.io.read_raw_fif(raw_fname)
        except ValueError:
            continue
        raw.load_data()
        raw.pick_types(eeg=True)

        # Compute SSD using subject's peak frequency
        alpha_bp = [peak - bin_width, peak + bin_width]
        noise_bp = [peak - (bin_width + 2), peak + (bin_width + 2)]
        noise_bs = [peak - (bin_width + 1), peak + (bin_width + 1)]
        try:
            filters, patterns = core.ssd.compute_ssd(
                raw, alpha_bp, noise_bp, noise_bs)
        except ValueError:
            continue

        # Save patterns and filters
        df = pd.DataFrame(filters.T, columns=raw.ch_names)
        df.to_csv(ssd_filters_fname, index=False)
        df = pd.DataFrame(patterns.T, columns=raw.ch_names)
        df.to_csv(ssd_patterns_fname, index=False)


if __name__ == '__main__':
    run_ssd_all_subjects()
