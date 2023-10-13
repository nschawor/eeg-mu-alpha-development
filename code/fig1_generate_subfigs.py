"""Generate subfigures for Figure 1A, 1C, 1D, and 1F.
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import gridspec
import seaborn as sns
import mne
from core.helper import load_ssd, get_subject_list
from core.params import *
import pyvista as pv
pv.rcParams['transparent_background'] = True


def plot_sensors(
        subj=EXAMPLE_SUBJ, mne_raw_folder=MNE_RAW_FOLDER,
        save_folder=FIG_FOLDER):
    """Plot sensors for a subject.

    Parameters
    ----------
    subj : str
        Subject to plot sensors for.
    mne_raw_folder : str
        Folder containing MNE raw files.
    save_folder : str
        Folder to save figure in.
    """
    # Read in electrode positions from a file
    fname_electrodes = f'{mne_raw_folder}/{subj}-raw.fif'
    raw = mne.io.read_raw_fif(fname_electrodes)

    # Make figure of sensors
    _, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    fig = raw.plot_sensors(ch_type='eeg', axes=ax, show=False)
    plt.tight_layout()

    # Save figure
    save_fname = f'{save_folder}/sensors_{subj}.pdf'
    fig.set_size_inches(5, 5)
    plt.savefig(save_fname, dpi=300, transparent=True)


def age_hist(
        subjects_csv=ALL_SUBJECTS_CSV, save_folder=FIG_FOLDER, age_max=AGE_MAX):
    """Plot histogram of ages for dataset.

    Parameters
    ----------
    subjects_csv : str
        CSV file containing ages for all subjects.
    save_folder : str
        Folder to save figure in.
    age_max : int
        Maximum age to plot.
    """
    # Load ages DataFrame
    subj_df = pd.read_csv(subjects_csv, index_col='EID')
    subj_df = subj_df.query('Age <= @age_max').round({'Age': 0})

    # Plot histogram
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    sns.histplot(
        data=subj_df, x='Age', discrete=True, legend=False,
        palette='mako_r', facecolor='w', alpha=0.8, ax=ax)
    plt.ylabel('subjects')
    ax.set(xlabel='age')
    sns.despine(ax=ax)

    # Save histogram
    save_fname = f'{save_folder}/age_hist.pdf'
    fig.set_size_inches(2/1.2, 2.25/1.2)
    plt.tight_layout()
    plt.savefig(save_fname, dpi=300, transparent=True)


def diagnoses_hist(
        diagnoses_csv=DIAGNOSES_CSV, diagnoses_to_plot=DIAGNOSES_TO_PLOT,
        save_folder=FIG_FOLDER):
    """Plot histogram of diagnoses for dataset.

    Parameters
    ----------
    diagnoses_csv : str
        CSV file containing diagnoses for all subjects.
    diagnoses_to_plot : dict
        Dictionary mapping diagnoses to plot to diagnoses in CSV file.
    save_folder : str
        Folder to save figure in.
    """
    # Load diagnoses DataFrame
    if not os.path.exists(diagnoses_csv):
        print('Cannot make diagnoses histogram without data being unzipped.')
        return

    diagnoses_df = pd.read_csv(diagnoses_csv, index_col=0)
    diagnoses_df = diagnoses_df.rename(columns={
        'subject': 'EID', 'diagnosis': 'Diagnosis'})
    diagnoses_df = diagnoses_df[diagnoses_df['Diagnosis'].isin(
        values=list(diagnoses_to_plot.keys()))]
    diagnoses_df['Diagnosis'] = diagnoses_df['Diagnosis'].map(
        diagnoses_to_plot)

    # Plot histogram
    diagnoses = sorted(list(set(diagnoses_to_plot.values())))
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
    sns.countplot(
        data=diagnoses_df, y='Diagnosis', order=diagnoses, width=1,
        palette='flare', saturation=0.8, facecolor='w', edgecolor='k', ax=ax)
    ax.set(xlabel='subjects', ylabel='diagnosis')
    sns.despine(ax=ax)
    fig.set_size_inches(2/1.2, 1.5/1.2)
    plt.tight_layout()

    # Save histogram
    save_fname = f'{save_folder}/diagnoses_hist.pdf'
    plt.savefig(save_fname, dpi=300, transparent=True)


def plot_ssd(
        subj=EXAMPLE_SUBJ, channel=EXAMPLE_CHANNEL,
        channels=tuple(LAPLACIAN_CHANNELS.keys()),
        mne_raw_folder=MNE_RAW_FOLDER, peak_freq_csv=PEAK_FREQ_CSV,
        save_folder=FIG_FOLDER):
    """Plot SSD for an example subject.

    Parameters
    ----------
    subj : str
        Subject to plot SSD for.
    channel : str
        Channel to plot SSD for.
    channels : tuple
        Set of channels that must contain specified channel to plot SSD for.
    mne_raw_folder : str
        Folder containing MNE raw files.
    peak_freq_csv : str
        CSV file containing peak frequencies for all subjects.
    save_folder : str
        Folder to save figure in.
    """
    assert channel in channels

    # Load raw for subject
    raw_fn = f'{mne_raw_folder}/{subj}-raw.fif'
    raw = mne.io.read_raw_fif(raw_fn)

    # Get peak frequency from CSV
    df = pd.read_csv(peak_freq_csv)
    peak = df[df.subject == subj].peak.values[0]

    # Compute PSD for selected channels of raw
    copy = raw.copy()
    copy.pick_channels([channel])
    psd_raw, freq_raw = mne.time_frequency.psd_welch(
        copy, fmin=1, fmax=45, n_fft=2001)

    # Plot PSD
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(freq_raw, psd_raw.flatten(), color='k')
    ax.axvline(peak, ls='--', c='k')
    ax.set(yscale='log')
    ax.set_xlim(right=30)
    ylim_bottom = ax.get_ylim()[0]

    # Highlight signal band
    cmap = get_cmap('bone')
    signal_bounds_idx = np.abs(freq_raw - (peak - 2)).argmin(), np.abs(
        freq_raw - (peak + 2)).argmin() + 1
    signal_x = freq_raw[slice(*signal_bounds_idx)]
    signal_y = psd_raw[0, slice(*signal_bounds_idx)]
    signal_power_min = np.repeat(ylim_bottom, len(
        freq_raw[slice(*signal_bounds_idx)]))
    ax.fill_between(
        signal_x, signal_y, signal_power_min, color=cmap(0.5), alpha=0.5,
        label='signal')

    # Highlight right noise band
    rnoise_bounds_idx = np.abs(freq_raw - (peak + 2)).argmin(), np.abs(
        freq_raw - (peak + 4)).argmin() + 1
    rnoise_x = freq_raw[slice(*rnoise_bounds_idx)]
    rnoise_y = psd_raw[0, slice(*rnoise_bounds_idx)]
    rnoise_power_min = np.repeat(ylim_bottom, len(
        freq_raw[slice(*rnoise_bounds_idx)]))
    ax.fill_between(
        rnoise_x, rnoise_y, rnoise_power_min, color=cmap(0.8), alpha=0.5)
    ax.set_ylim(bottom=ylim_bottom)

    # Highlight left noise band
    lnoise_bounds_idx = np.abs(freq_raw - (peak - 4)).argmin(), np.abs(
        freq_raw - (peak - 2)).argmin() + 1
    lnoise_x = freq_raw[slice(*lnoise_bounds_idx)]
    lnoise_y = psd_raw[0, slice(*lnoise_bounds_idx)]
    lnoise_power_min = np.repeat(ylim_bottom, len(
        freq_raw[slice(*lnoise_bounds_idx)]))
    ax.fill_between(
        lnoise_x, lnoise_y, lnoise_power_min, color=cmap(0.8), alpha=0.5,
        label='noise')

    # Draw arrow to signal desire to increase power in signal band, ensuring
    # that placement is good on semilog plot
    signal_arrow_base = (np.mean(signal_y) * ylim_bottom ** 3) ** (1/4)
    signal_arrow_end = (np.mean(signal_y) ** 3 * ylim_bottom) ** (1/4)
    signal_arrow_len = signal_arrow_end - signal_arrow_base
    signal_head_base = (signal_arrow_base * signal_arrow_end ** 4) ** (1/5)
    signal_head_len = signal_arrow_end - signal_head_base
    ax.arrow(
        np.mean(signal_x), signal_arrow_base, 0, signal_arrow_len, width=0.75,
        length_includes_head=True, head_length=signal_head_len, head_width=2,
        color=cmap(0.25), ec=cmap(0.0), zorder=4)

    # Draw arrows to signal desire to decrease power in noise band, ensuring
    # that placement is good on semilog plot
    rnoise_arrow_base = (np.mean(rnoise_y) ** 3 * ylim_bottom) ** (1/4)
    rnoise_arrow_end = (np.mean(rnoise_y) * ylim_bottom ** 3) ** (1/4)
    rnoise_arrow_len = rnoise_arrow_end - rnoise_arrow_base
    rnoise_head_base = (rnoise_arrow_base * rnoise_arrow_end ** 4) ** (1/5)
    rnoise_head_len = rnoise_head_base - rnoise_arrow_end
    ax.arrow(
        np.mean(rnoise_x), rnoise_arrow_base, 0, rnoise_arrow_len, width=0.4,
        length_includes_head=True, head_length=rnoise_head_len, head_width=0.75,
        color=cmap(0.25), ec=cmap(0.0))
    ax.arrow(
        np.mean(lnoise_x), rnoise_arrow_base, 0, rnoise_arrow_len, width=0.4,
        length_includes_head=True, head_length=rnoise_head_len, head_width=0.75,
        color=cmap(0.25), ec=cmap(0.0))

    # Plot aesthetics
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('log power')
    ax.tick_params(axis='both', which='major')
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    sns.despine(ax=ax)

    # Save figure
    save_fname = f'{save_folder}/ssd.pdf'
    fig.set_size_inches(2.5, 3)
    plt.tight_layout()
    plt.savefig(save_fname, dpi=300, transparent=True)


def plot_leadfield(
        subj=EXAMPLE_SUBJ, comp=EXAMPLE_COMP, regions=EXAMPLE_REGIONS,
        region_colors=(MU_COLOR, ALPHA_COLOR), mri_folder=MRI_FOLDER,
        mne_raw_folder=MNE_RAW_FOLDER, parcellation_subj=PARCELLATION_SUBJ,
        annotation=PARCELLATION_ANNOTATION, save_folder=FIG_FOLDER):
    """Plot leadfield for a component.

    Parameters
    ----------
    subj : str
        Subject to plot leadfield for.
    comp : int
        Component to plot leadfield for.
    regions : tuple
        Regions to plot leadfield for.
    region_colors : tuple
        Colors for regions.
    mri_folder : str
        Folder containing MRI files.
    mne_raw_folder : str
        Folder containing MNE raw files.
    parcellation_subj : str
        Subject to use for parcellation.
    annotation : str
        Annotation to use for parcellation.
    save_folder : str
        Folder to save figure in.
    """
    # Load forward solution
    fwd_fname = f"{mri_folder}/eeg_fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)

    # Load raw file
    raw_fn = f'{mne_raw_folder}/{subj}-raw.fif'
    raw = mne.io.read_raw_fif(raw_fn)
    raw.load_data()
    raw.pick_types(eeg=True)

    # Load pattern
    patterns, _ = load_ssd(subj)
    pattern = patterns[:, comp]

    fig, ax = plt.subplots()
    mne.viz.plot_topomap(pattern, raw.info, axes=ax, show=False)
    fig.set_size_inches(2, 2)
    fig.savefig(f'{save_folder}/test.pdf', dpi=300, transparent=True)

    # Initialize figure
    fig, axes = plt.subplots(figsize=(20, 8))
    height_ratios = [1] + [4] * len(regions)
    width_ratios = [12, 1, 3, 4, 3]
    gs = gridspec.GridSpecFromSubplotSpec(
        len(regions)+1, 5, subplot_spec=axes, height_ratios=height_ratios,
        width_ratios=width_ratios)
    axes.axis('off')

    # Load labels and isolate desired regions
    labels = mne.read_labels_from_annot(
        parcellation_subj, annotation, 'lh', subjects_dir=mri_folder,
        verbose=False)
    labels = [label for label in labels if label.name in regions]
    labels = list(np.array(labels)[np.argsort(regions)])

    # Initialize brain
    brain = mne.viz.Brain(
        parcellation_subj, 'lh', 'inflated', subjects_dir=mri_folder,
        cortex='low_contrast', background='white', size=(800, 600))

    # Enable text to be printed in LaTeX apprioriately
    plt.rcParams.update({
        'text.usetex': True, 'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica']})

    # Add titles to lead fields and spatial patterns
    ax_lf_title = fig.add_subplot(gs[0, -4:-2])
    ax_lf_title.axis('off')
    ax_lf_title.text(
        0.5, 0.5, 'Lead fields', fontsize=24, va='center', ha='center')
    ax_pattern_title = fig.add_subplot(gs[0, -1])
    ax_pattern_title.axis('off')
    ax_pattern_title.text(
        0.5, 0.5, 'Spatial pattern', fontsize=24, va='center', ha='center')

    # Iterate over desired regions
    for i, (label, color) in enumerate(zip(labels, region_colors)):
        # Determine center of mass in desired region
        center_of_mass = label.center_of_mass(
            subject=parcellation_subj, surf='inflated', subjects_dir=mri_folder)

        # Plot foci from desired regions on 3d brain
        brain.add_foci(
            center_of_mass, coords_as_verts=True, hemi='lh', color=color,
            scale_factor=1.5)

        # Place lead field index
        fig, ax_lf = plt.subplots()
        ax_lf.axis('off')

        # Plot matched leadfield right of brain
        center_of_mass_coords = mne.vertex_to_mni(
            center_of_mass, 0, parcellation_subj, subjects_dir=mri_folder)
        source_coords = mne.head_to_mni(
            fwd['source_rr'], parcellation_subj, fwd['mri_head_t'],
            subjects_dir=mri_folder)
        source_idx = np.linalg.norm(
            source_coords - center_of_mass_coords, axis=1).argmin()
        lf = fwd['sol']['data'][:, source_idx]
        mne.viz.plot_topomap(lf, raw.info, axes=ax_lf, show=False)
        fig.set_size_inches(2, 2)
        fig.savefig(
            f'{save_folder}/topo_{label.name}.pdf', dpi=300, transparent=True)

    # Plot example component on far right
    ax_comp = fig.add_subplot(gs[1:, -1])
    mne.viz.plot_topomap(pattern, raw.info, axes=ax_comp, show=False)

    # Take screenshot to add to plot
    fig, ax = plt.subplots()
    img = brain.screenshot()
    # ax_sources = fig.add_subplot(gs[:, 0])
    ax.imshow(img[115:495, 125:675, :])
    ax.axis('off')

    # Close brain
    brain.close()

    # Save figure
    save_fname = f'{save_folder}/leadfield.pdf'
    fig.set_size_inches(5, 5)
    fig.savefig(save_fname, dpi=300, transparent=True)

    # Revert RC params to default
    plt.rcParams.update(plt.rcParamsDefault)


def plot_snr_by_prop_subjs_with_rhythm(
        snr_threshold=SNR_THRESHOLD, alpha_color=ALPHA_COLOR, mu_color=MU_COLOR,
        sources_csv=SOURCES_CSV, save_folder=FIG_FOLDER):
    """Plot SNR by proportion of subjects with rhythm.

    Parameters
    ----------
    snr_threshold : float
        SNR threshold to plot as vertical line.
    alpha_color : str
        Color for alpha rhythm.
    mu_color : str
        Color for mu rhythm.
    sources_csv : str
        CSV file containing sources for all subjects.
    save_folder : str
        Folder to save figure in.
    """
    # Get sources for all subjects
    sources_df = pd.read_csv(sources_csv)
    sources_df = sources_df[sources_df.pattern_dist < PATTERN_DIST_THRESHOLD] 

    # Determine the number of subjects
    # n_subjs = len(get_subject_list('file'))
    df = pd.read_csv('../results/subj_dropout.csv')
    n_subjs = df.iloc[1]['Number of subjects remaining']

    # Create steps of SNR
    snr_max = np.ceil(sources_df['SNR'].max())
    snr_steps = np.arange(0, snr_max, 0.5)

    # Initialize plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    rhythm_colors = {'alpha': alpha_color, 'mu': mu_color}
    plateau_pts = []
    for rhythm, color in rhythm_colors.items():
        # Extract component of particular rhythm type
        rhythm_df = sources_df.query(rhythm)

        # Take only the highest SNR component for each subject
        rhythm_df = rhythm_df.groupby('subject').min(['idx']).reset_index()

        # Bin by SNR
        out = pd.cut(rhythm_df['SNR'], snr_steps)

        # Count the number in each bin after applying threshold and determine
        # proportion of subjects
        counts = out.value_counts(sort=False)[::-1].cumsum()[::-1]
        pcts = counts / n_subjs

        # Plot proportions
        ax.plot(
            snr_steps[:-1], pcts.values*100, color=color, label=rhythm, lw=2.5)

        # Determine SNR where plot plateaus for rhythm
        plateau_pt = np.where(np.diff(pcts.values) == 0)[0][0]
        plateau_pts.append(plateau_pt)

    # Add used SNR threshold as vertical line to plot
    ax.axvline(snr_threshold, c='gray', ls='--', zorder=-3)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_yticklabels([f'{a}%' for a in ax.get_yticks()])

    # Plot aesthetics
    # ax.set(xticks=[0, 5, 10, 20])
    ax.set_xticks([0, 5, 10, 20])
    ax.set_xlim(0, 20) #snr_steps[np.max(plateau_pts)])
    ax.set_xlabel('1/f-corrected SNR [dB]')
    ax.set_ylabel('proportion of subjects\nwith rhythm')
    plt.legend()
    fig.set_size_inches(2.5, 2)
    fig.tight_layout()
    sns.despine(fig)

    # Save figure
    save_fname = f'{save_folder}/snr_by_prop_subjs_with_rhythm.pdf'
    fig.tight_layout()
    fig.savefig(save_fname, dpi=300, transparent=True)

# %%
if __name__ == '__main__':
    plt.style.use('figures.mplstyle')

    # Make sure figure folder exists
    os.makedirs(FIG_FOLDER, exist_ok=True)

    # # Plot sensors
    plot_sensors()

    # # Create histogram for ages
    age_hist()

    # # Make SSD plot
    plot_ssd()

    # # Make plot for leadfield
    plot_leadfield()

    # Make plot for SNR by proportion of subjects with rhythm
    plot_snr_by_prop_subjs_with_rhythm()


# %%
