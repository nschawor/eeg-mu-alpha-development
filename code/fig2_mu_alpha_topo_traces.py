"""Plot traces and average topomaps for alpha and mu rhythms (Figure 2A).
"""
import pandas as pd
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import core.ssd
from core.helper import get_alpha_mu_comps
from core.params import *


def percentile_spectrum(
        raw, band=(8, 12), nr_lines=5, i_chan=0, nr_seconds=4):
    """Function to compute the percentile spectrum: Cut the given signal into
    segments of [nr_seconds] length. Compute PSD for each segment. Sort the
    segments according to the power in a given frequency band. Divide into
    percentile groups and compute the average PSD for each group.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object containing the EEG data.
    band: tuple
        Frequency band for sorting the PSDs.
    nr_lines: int
        Number of groups.
    i_chan: int
        Channel index.
    nr_seconds: int
        Segment length.

    Returns
    -------
    psd_perc : array
        Percentile spectrum.
    freq : array
        Frequency axis of computed spectrum.
    idx_segments : array
        Indices of segments sorted by power in the given frequency band.
    """
    # Create events and Epochs
    events = mne.make_fixed_length_events(
        raw, start=0, stop=raw.times[-1], duration=nr_seconds)
    epo = mne.Epochs(raw, events, tmin=0, tmax=nr_seconds, baseline=None)

    # Compute PSD
    n_fft = nr_seconds * int(raw.info['sfreq'])
    psd, freq = mne.time_frequency.psd_welch(
        epo, picks=[i_chan], fmin=1, fmax=45, n_fft=n_fft)

    # Sort segments by power in the given frequency band
    idx_start = np.argmin(np.abs(freq - band[0]))
    idx_end = np.argmin(np.abs(freq - band[1]))
    mean_power = np.mean(psd[:, 0, idx_start:idx_end], axis=1)
    idx_segments = np.argsort(mean_power, axis=0)[::-1]
    spacing = int(np.floor(len(events)/nr_lines))

    # Compute percentiles
    psd_perc = np.zeros((nr_lines, psd.shape[-1]))
    for i in range(nr_lines):
        idx1 = idx_segments[i * spacing:(i + 1) * spacing]
        psd_perc[i] = np.mean(psd[idx1, 0, :], axis=0)
    return psd_perc, freq, idx_segments


if __name__ == '__main__':
    # Get alpha and mu components
    alpha_comps_idx, mu_comps_idx = get_alpha_mu_comps()

    for i_type, comps_idx in enumerate([alpha_comps_idx, mu_comps_idx]):
        fig, ax = plt.subplots(1, 1)
        # Filter out components with bad SNR
        df = comps_idx.query("SNR > @SNR_THRESHOLD")

        # Filter out components without good enough fit to leadfield
        df = df.query("pattern_dist < @PATTERN_DIST_THRESHOLD")
        subjects = df.subject.to_list()
        df = df.set_index('subject')

        #% Collect all patterns for rhythm across subjects
        patterns_all = np.zeros((111, len(subjects)))
        pats = []
        for subj in subjects:
            # Load patterns for subject
            pat = pd.read_csv(f'../results/ssd/{subj}_ssd_patterns.csv')
            comp = df.loc[subj]
            if type(df.loc[subj]) != pd.core.series.Series:
                comp = comp.iloc[0]
            idx = int(comp.idx)
            topo = pat.iloc[idx].to_numpy()
            idx_max = np.argmax(np.abs(topo))
            pats.append(pat.iloc[idx]*np.sign(topo[idx_max]))

        #% Get average pattern for rhythm across subjects
        patt = pd.concat(pats, axis=1).T
        average_pattern = np.mean(patt, axis=0)

        #% Load electrode positions
        raw = mne.io.read_raw_fif('../data/electrode_positions-raw.fif')
        for i in range(len(raw.info['chs'])):
            raw.info['chs'][i]['loc'] /= 100

        # Plot average topomap
        mne.viz.plot_topomap(average_pattern, raw.info, axes=ax, show=False)
        fig.set_size_inches(2, 2)
        fig.savefig(f'avg_topo_{i_type}.pdf', dpi=300, transparent=True)

    # %% Plot traces for the 20 best subjects and then select one
    plt.style.use('figures.mplstyle')
    conditions = {'alpha': alpha_comps_idx, 'mu': mu_comps_idx}
    colors = {'alpha': ALPHA_COLOR, 'mu': MU_COLOR}

    for i_cond, cond in enumerate(conditions):
        print(cond)
        comp = conditions[cond]
        comp = comp[comp.pattern_dist < PATTERN_DIST_THRESHOLD]
        comp = comp.sort_values('SNR', ascending=False)

        # Select 20 best subjects
        for i_sub in range(20):
            subject = comp.iloc[i_sub].subject
            idx = comp.iloc[i_sub].idx

            # Determine file names for necessary files
            ssd_filters_fname = f'{SSD_FOLDER}/{subject}_ssd_filters.csv'
            ssd_patterns_fname = ssd_filters_fname.replace(
                'filters', 'patterns')
            raw_fname = f'{MNE_RAW_FOLDER}/{subject}-raw.fif'

            # Load the raw file
            raw = mne.io.read_raw_fif(raw_fname)
            raw.load_data()
            raw.pick_types(eeg=True)

            # Load SSD filters and patterns
            filters = pd.read_csv(
                ssd_filters_fname).values.T[:, :NUM_COMPONENTS]
            patterns = pd.read_csv(
                ssd_patterns_fname).values.T[:, :NUM_COMPONENTS]

            # Apply filters
            raw_ssd = core.ssd.apply_filters(raw, filters)
            raw_ssd.filter(1, 45)
            nr_seconds = 1
            psd_perc, freq, idx_segments = percentile_spectrum(
                raw, band=(8, 12), nr_lines=5, i_chan=idx,
                nr_seconds=nr_seconds)

            # Plot traces
            fig, ax = plt.subplots()
            fig.set_size_inches(2.5, 3.5)
            epo = mne.make_fixed_length_epochs(raw_ssd, duration=nr_seconds)
            epo.load_data()
            for ii, i in enumerate(idx_segments[2:6]):
                sig = epo._data[i, idx]
                sig /= np.ptp(sig)
                ax.plot(epo.times, sig+ii, color=colors[cond], lw=1)
            ax.set(xlim=(epo.times[0], epo.times[-1]), yticks=[], xticks=[0,1],
                   xlabel='time [s]')
            sns.despine(fig, left=True)
            fig.savefig(f'traces_{cond}_{i_sub}_{subject}.pdf', dpi=200)
            fig.show()
