"""Make PSD plots for two occipital and two central sensors for one subject
(Figure 1B).
"""
#%%
import mne
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from core.helper import load_ssd, load_comp_timeseries
from core.params import *

# Set colors
colors = plt.cm.tab20(np.linspace(0, 1, 20))
alpha_c = colors[6:8]
mu_c = colors[4:6]


def psd_one_subj(
        raw, peak_freq_df, subj, ax, alpha_colors=alpha_c, mu_colors=mu_c):
    """Plot PSD for one example subject.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data for subject.
    peak_freq_df : pd.DataFrame
        DataFrame with peak frequencies for all subjects.
    subj : str
        Subject ID.
    ax : tuple of matplotlib.axes.Axes
        Axes to plot on.
    alpha_colors : list of matplotlib.colors
        Colors for alpha PSDs.
    mu_colors : list of matplotlib.colors
        Colors for mu PSDs.

    Returns
    -------
    fig2 : matplotlib.figure.Figure
        Figure with PSD plots.
    """
    # Compute PSD for selected channels of raw
    channels = ['E70', 'E83', 'E36', 'E104']
    copy = raw.copy()
    raw.pick_types(eeg=True)

    # Pick channels
    picks = mne.pick_channels(raw.ch_names, channels, ordered=True)
    print(len(raw.ch_names))
    print(picks)

    # Plot sensors
    fig2 = raw.plot_sensors()
    print(copy.ch_names)

    # Compute PSD for channels
    copy.pick_channels(channels, ordered=True)
    psd_raw, freq_raw = mne.time_frequency.psd_welch(
        copy, fmin=1, fmax=45, n_fft=2001)

    # Plot PSD for channels
    colors_dict = {ch: c for ch, c in zip(
        channels, [*alpha_colors[:2], *mu_colors[:2]])}
    colors = [colors_dict.get(i, i) for i in copy.ch_names]
    ch_flat = [copy.ch_names[i] for i in np.repeat(np.arange(
        psd_raw.shape[0]), psd_raw.shape[1])]
    df_raw = pd.DataFrame({
        'Frequency': np.tile(freq_raw, psd_raw.shape[0]),
        'Power': psd_raw.flatten(), 'Channel': ch_flat})
    sns.lineplot(data=df_raw, x='Frequency', y='Power', hue='Channel',
                 ax=ax[0], palette=colors)
    ax[0].set(xscale='linear', yscale='log')
    ax[0].legend(title='channel', title_fontsize=10, fontsize=10)
    ax[0].set_xlabel('frequency [Hz]')
    ax[0].set_ylabel('log power')
    ax[0].tick_params(axis='both', which='major')

    # Plot extracted alpha-band peak frequency for subject
    peak_freq = peak_freq_df[peak_freq_df['subject'] == subj]['peak'].values
    ax[0].axvline(peak_freq, ls='--', c='k')
    ax[0].text(
        peak_freq + 1, np.min(df_raw['Power']),
        f'Fit alpha-band \npeak = {peak_freq[0]:.3f} Hz', ha='left')
    sns.despine(ax=ax[0])
    return fig2

if __name__ == '__main__':
    #%%
    peak_freq_csv=PEAK_FREQ_CSV
    peak_freq_df = pd.read_csv(peak_freq_csv)

    subj = 'NDARNK489GNR'
    raw_fn = f'{MNE_RAW_FOLDER}/{subj}-raw.fif'

    # Load raw file
    raw = mne.io.read_raw_fif(raw_fn)
    raw.load_data()
    raw.pick_types(eeg=True)

    # Load SSD file
    patterns, filters = load_ssd(subj)

    # Load component time series
    ts, _ = load_comp_timeseries(subj)

    # Create figure for all plots
    #%%
    plt.style.use('figures.mplstyle')
    fig = plt.figure(figsize=(3, 3))

    # Outer gridspec, upper half for PSDs and bottom half for patterns
    gs = gridspec.GridSpec(1, 1, figure=fig, top=0.9, bottom=0.15, hspace=0.3)

    # Plot PSDs
    ax_psd_raw = fig.add_subplot(gs[0, 0])
    fig2 = psd_one_subj(
            raw, peak_freq_df, subj, (ax_psd_raw, None, None))
    idx = [63, 74, 34, 92]
    colorr = np.ones((111, 4))
    colorr[:,:3] = 0
    colorr[idx[:2]] = alpha_c
    colorr[idx[2:]] = mu_c

    fig2.axes[0].get_children()[5].set_edgecolors(colorr)
    fig2.axes[0].get_children()[5].set_facecolors(colorr)
    sizes = np.array([20]*111)
    sizes[idx] = 40
    fig2.axes[0].get_children()[5].set_sizes(sizes)
    save_fname = f'{FIG_FOLDER}/raw_psd_example_sens.pdf'
    fig2.set_size_inches(3, 3)
    fig2.savefig(save_fname, dpi=300, transparent=True)
    save_fname = f'{FIG_FOLDER}/raw_psd_example.pdf'
    fig.savefig(save_fname, dpi=300, transparent=True)
    # %%
