"""Relatively general-purpose functions used in multiple scripts.
"""
# Import packages
import os.path
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import fooof
import pingouin as pg
from statannotations.Annotator import Annotator
import core.ssd as ssd
from core.params import *
import scipy.linalg
import numpy as np
import pyvista as pv


def get_subject_list(
        pipeline_step='bycycle', all_subjects_csv=ALL_SUBJECTS_CSV,
        mne_raw_folder=MNE_RAW_FOLDER, results_folder=RESULTS_FOLDER,
        ssd_folder=SSD_FOLDER, ssd_sparam_folder=SSD_SPARAM_FOLDER,
        bycycle_folder=BYCYCLE_FOLDER, age=None) -> list:
    """Convenience function for subselectrion of subjects according to specific
    criteria.

    Parameters
    ----------
    pipeline_step : str
        Pipeline step to subselect subjects for. Options are 'file', 'sparam',
        'ssd', 'ssd_sparam', 'sources', 'bycycle'.
    all_subjects_csv : str
        Path to CSV file containing all subjects.
    mne_raw_folder : str
        Path to folder containing raw MNE files.
    results_folder : str
        Path to folder containing results.
    ssd_folder : str
        Path to folder containing SSD results.
    ssd_sparam_folder : str
        Path to folder containing SSD spectral parameterization results.
    bycycle_folder : str
        Path to folder containing bycycle results.
    age : int | None
        Age to subselect subjects for.

    Returns
    -------
    subjects : list
        List of subjects that meet the criteria.
    """
    df_all = pd.read_csv(all_subjects_csv)
    subjects = df_all.EID

    # Make booleans from pipeline step
    steps = ['file', 'sparam', 'ssd', 'ssd_sparam', 'sources', 'bycycle']
    assert pipeline_step in steps
    step_given = steps.index(pipeline_step)
    step_bools = {step: i <= step_given for i, step in enumerate(steps)}

    # return subjects with specific age
    if age:
        df_age = df_all[df_all.Age.apply(np.floor) == age]
        print(f'Number of subjects with age between {age} and {age + 1}: '
              f'{len(df_age)}')
        subjects = [subj for subj in subjects if subj in df_age.EID.to_list()]

    # check if the required files are present
    if step_bools['file']:
        subjects = [subj for subj in subjects if os.path.exists(
            f'{mne_raw_folder}/{subj}-raw.fif')]

    if step_bools['sparam']:
        csv_fn = f'{results_folder}/subjects_with_peak_frequency.csv'
        df = pd.read_csv(csv_fn)
        subjects = [subj for subj in subjects if subj in df.subject.to_list()]

    if step_bools['ssd']:
        subjects = [subj for subj in subjects if os.path.exists(
            f'{ssd_folder}/{subj}_ssd_filters.csv')]

    if step_bools['ssd_sparam']:
        subjects = [subj for subj in subjects if os.path.exists(
            f'{ssd_sparam_folder}/{subj}_patterns.csv')]

    if step_bools['sources'] and os.path.exists(SOURCES_CSV):
        alpha_comps, mu_comps = get_alpha_mu_comps()
        subjects = [subj for subj in subjects if subj in set(
            pd.concat((alpha_comps, mu_comps))['subject'])]

    if step_bools['bycycle']:
        subjects = [subj for subj in subjects if os.path.exists(
            f'{bycycle_folder}/{subj}.csv')]

    subjects = sorted(subjects)
    return subjects


def plot_3d_brain(plotter, mri_folder=MRI_FOLDER):
    """Creates Pyvista object for Freesurfer 3D brain.

    Parameters
    ----------
    plotter : instance of Plotter
        Pyvista plotter object.
    mri_folder : str
        Path to folder containing MRI data

    Returns
    -------
    plotter : instance of Plotter
        Pyvista plotter object with brain added.
    """
    # Read trans file
    trans_fname = f"{mri_folder}/fonov-trans.fif"
    trans = mne.read_trans(trans_fname)

    # Load inner skull
    fname = f'{mri_folder}/fonov/bem/inner_skull.surf'
    pos, tri = mne.read_surface(fname)
    pos = pos/1000

    # Apply transform to inner skull
    pos = mne.transforms.apply_trans(scipy.linalg.pinv(trans['trans']), pos)

    # Create mesh and add to plotterß
    faces = np.hstack((3 * np.ones((tri.shape[0], 1)), tri))
    faces = faces.astype("int")
    brain_cloud = pv.PolyData(pos, faces)
    plotter.add_mesh(brain_cloud, opacity=0.1, style='wireframe')

    # Apply transform to each hemisphere
    for hemi in ['lh', 'rh']:
        # Load surface
        fname = f'{mri_folder}/fonov/surf/{hemi}.white'
        pos, tri = mne.read_surface(fname)
        pos = pos/1000

        # Apply transform to surface
        pos = mne.transforms.apply_trans(scipy.linalg.pinv(trans['trans']), pos)

        # Create mesh and add to plotter
        faces = np.hstack((3 * np.ones((tri.shape[0], 1)), tri))
        faces = faces.astype("int")
        brain_cloud = pv.PolyData(pos, faces)
        plotter.add_mesh(brain_cloud, opacity=0.5)
    return plotter


def get_source_labels(
        annotation=PARCELLATION_ANNOTATION, subject=PARCELLATION_SUBJ,
        mri_folder=MRI_FOLDER, names=True):
    """Label alpha and mu sources.

    Parameters
    ----------
    annotation : str
        Name of annotation file for desired parcellation.
    subject : str
        Name of subject for desired parcellation.
    mri_folder : str
        Path to folder containing MRI data.
    names : bool
        If True, returns names of labels instead of label objects.

    Returns
    -------
    alpha_selected : list
        List of labels for alpha sources.
    mu_selected : list
        List of labels for mu sources.
    """
    labels = mne.read_labels_from_annot(
        subject, annotation, 'both', subjects_dir=mri_folder, verbose=False)

    # file is from:
    # https://neuroimaging-core-docs.readthedocs.io/en/latest/pages/atlases.html
    # contains data frame with classifications for each region
    df = pd.read_csv(f'{DATA_FOLDER}/HCP-MMP1_UniqueRegionList.csv')

    # select visual labels
    labels_roi = df.query('Lobe=="Occ"').regionName
    labels_roi = ['_'.join(l.split('_')[::-1]) for l in labels_roi]
    alpha_selected = []
    for labels_roi1 in labels_roi:
        for label in labels:
            if label.name.startswith(labels_roi1 + '_'):
                if names:
                    alpha_selected.append(label.name)
                    continue
                alpha_selected.append(label)

    assert len(labels_roi) == len(alpha_selected)

    # select somatosensory labels
    labels_roi = df.query('cortex == "Somatosensory_and_Motor" or cortex == '
                          '"Paracentral_Lobular_and_Mid_Cingulate" or cortex '
                          '== "Premotor"').regionName
    labels_roi = ['_'.join(l.split('_')[::-1]) for l in labels_roi]
    mu_selected = []
    for labels_roi1 in labels_roi:
        for label in labels:
            if label.name.startswith(labels_roi1 + '_'):
                if names:
                    mu_selected.append(label.name)
                    continue
                mu_selected.append(label)

    assert len(labels_roi) == len(mu_selected)
    return alpha_selected, mu_selected


def apply_laplacian(raw, channels):
    """Use a Laplacian over desired channels using each channel's neighbors.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance to apply Laplacian to.
    channels : dict
        Keys are picked channels, while values are lists of neighboring
        channels.

    Returns
    -------
    raw : instance of Raw
        Raw instance with Laplacian applied to picked channels.

    """
    # Make copy of original to prevent mix-ups
    orig_raw = raw.copy()

    # Subtract average of neighbors from each picked channel
    ch_lst, processed_data = [], []
    for ch, neighbors in channels.items():
        ch_data = orig_raw.get_data(picks=ch)
        neighbor_data = orig_raw.get_data(picks=neighbors)

        processed_data.append(ch_data - np.mean(neighbor_data, axis=0))
        ch_lst.append(ch)

    # Concatenate all processed channel data
    processed_arr = np.concatenate(processed_data)

    # Make Raw instance from data
    info = orig_raw.pick_channels(ch_lst).info
    processed_raw = mne.io.RawArray(processed_arr, info)
    return processed_raw


def load_ssd(participant_id, results_folder=RESULTS_FOLDER):
    """Load spatial filters and patterns for a specific dataset.

    Parameters
    ----------
    participant : str
        Participant ID.
    results_folder : str
        Path to folder containing results.

    Returns
    -------
    patterns : np.ndarray, 2D
        Spatial patterns as computed by SSD.
    filters : np.ndarray, 2D
        Spatial filters as computed by SSD.
    """
    # Load filters
    ssd_filters_fname = f"{results_folder}/ssd/{participant_id}_ssd_filters.csv"
    filters_df = pd.read_csv(ssd_filters_fname)
    filters = filters_df.values.T

    # Load patterns
    ssd_patterns_fname = ssd_filters_fname.replace('filters', 'patterns')
    patterns_df = pd.read_csv(ssd_patterns_fname)
    patterns = patterns_df.values.T
    return patterns, filters


def get_SNR(raw, fmin=1, fmax=55, seconds=3, freq=[8, 13]):
    """Compute power spectrum and calculate 1/f-corrected SNR in one band.

    Parameters
    ----------
    raw : instance of Raw
        Raw instance containing traces for which to compute SNR
    fmin : float
        minimum frequency that is used for fitting spectral model.
    fmax : float
        maximum frequency that is used for fitting spectral model.
    seconds: float
        Window length in seconds, converts to FFT points for PSD calculation.
    freq : list
        SNR in that frequency window is computed.

    Returns
    -------
    SNR : array, 1D
        Contains SNR (1/f-corrected, for a chosen frequency) for each channel.
    """
    # Compute PSD
    SNR = np.zeros((len(raw.ch_names),))
    n_fft = int(seconds * raw.info["sfreq"])
    psd, freqs = mne.time_frequency.psd_welch(
        raw, fmin=fmin, fmax=fmax, n_fft=n_fft)

    # Fit FOOOF
    fm = fooof.FOOOFGroup()
    fm.fit(freqs, psd)

    # Compute aperiodic-corrected SNR
    for pick in range(len(raw.ch_names)):
        psd_corr = 10 * np.log10(psd[pick]) - 10 * fm.get_fooof(pick)._ap_fit
        idx = np.where((freqs > freq[0]) & (freqs < freq[1]))[0]
        idx_max = np.argmax(psd_corr[idx])
        SNR[pick] = psd_corr[idx][idx_max]
    return SNR


def load_raw_eeg(subj, mne_raw_folder=MNE_RAW_FOLDER):
    """Load raw EEG data for a specific subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    mne_raw_folder : str
        Path to folder containing raw MNE files.

    Returns
    -------
    raw : instance of Raw
        Raw instance containing EEG data for a specific subject.
    """
    # Determine file name of raw EEG
    raw_fname = f'{mne_raw_folder}/{subj}-raw.fif'

    # Load raw
    raw = mne.io.read_raw_fif(raw_fname)
    return raw


def load_comp_timeseries(
        subj, mne_raw_folder=MNE_RAW_FOLDER,
        save_folder=COMP_TIMESERIES_FOLDER, num_components=NUM_COMPONENTS,
        f_bandpass=F_BANDPASS, s_freq=S_FREQ):
    """Load component time series for a specific subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    mne_raw_folder : str
        Path to folder containing raw MNE files.
    save_folder : str
        Path to folder where component time series should be saved.
    num_components : int
        Number of components to load.
    f_bandpass : tuple
        Bandpass filter to apply to components (fmin, fmax).
    s_freq : float
        Sampling frequency of raw EEG data.

    Returns
    -------
    comps : np.ndarray, 2D
        Component time series with shape (n_components, n_timepoints).
    s_freq : float
        Sampling frequency of component time series.
    """
    # Make folder to save if necessary
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Load component time series from save if already saved seperately
    save_fname = f'{save_folder}/{subj}.npy'
    if os.path.exists(save_fname):
        return np.load(save_fname), s_freq

    # Determine filenames for SSD, raw, and figure to be saved
    raw_fn = f'{mne_raw_folder}/{subj}-raw.fif'

    # Load raw file
    raw = mne.io.read_raw_fif(raw_fn)
    raw.load_data()
    fs = raw.info['sfreq']
    raw.pick_types(eeg=True)

    # Load SSD file
    _, filters = load_ssd(subj)

    # Apply filters to get components
    comps = ssd.apply_filters(raw, filters).get_data()

    # Take only the components worth investigating
    if comps.shape[0] > num_components:
        comps = comps[:num_components, :]

    # Bandpass filter components
    comps = mne.filter.filter_data(
        comps, fs, *f_bandpass, n_jobs=mp.cpu_count())

    # Save components to speed up processing next time
    np.save(save_fname, comps)
    return comps, fs


def get_alpha_mu_comps(sources_csv=SOURCES_CSV):
    """Extract alpha and mu components from sources CSV.

    Parameters
    ----------
    sources_csv : str
        Path to CSV containing source localization results.

    Returns
    -------
    alpha_comps : pd.DataFrame
        DataFrame containing alpha components.
    mu_comps : pd.DataFrame
        DataFrame containing mu components.
    """
    # Read in CSV of pattern distances
    df = pd.read_csv(sources_csv)

    # Seperate out mu and alpha components
    alpha_comps = df.query("alpha")[['subject', 'idx', 'SNR', 'pattern_dist']]
    mu_comps = df.query("mu")[['subject', 'idx', 'SNR', 'pattern_dist']]
    return alpha_comps, mu_comps


def filter_by_thresholds(
        df, snr_threshold=SNR_THRESHOLD,
        pattern_dist_threshold=PATTERN_DIST_THRESHOLD):
    """Filter out components with bad SNR or high pattern distance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing waveform features.
    snr_threshold : float
        Threshold for SNR. Components with SNR below this threshold will be
        removed.
    pattern_dist_threshold : float
        Threshold for pattern distance. Components with pattern distance above
        this threshold will be removed.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing waveform features for components that meet the
        thresholds.
    """
    # Filter out components with bad SNR
    df = df.query("SNR > @snr_threshold")

    # Filter out components without good enough fit to leadfield
    df = df.query("pattern_dist < @pattern_dist_threshold")
    return df


def load_waveform_features(
        save_fname=WAVEFORM_FEATS_CSV, snr_threshold=SNR_THRESHOLD,
        pattern_dist_threshold=PATTERN_DIST_THRESHOLD):
    """Load waveform features for all subjects and apply thresholds. Note: this
    function will only work if proc_7 has been run and the waveform features CSV
    has been created.

    Parameters
    ----------
    save_fname : str
        Path to CSV containing waveform features.
    snr_threshold : float
        Threshold for SNR. Components with SNR below this threshold will be
        removed.
    pattern_dist_threshold : float
        Threshold for pattern distance. Components with pattern distance above
        this threshold will be removed.

    Returns
    -------
    all_waveform_feats : pd.DataFrame
        DataFrame containing waveform features for components that meet the
        thresholds.
    """
    # Load waveform features CSV if created
    if os.path.exists(save_fname):
        all_waveform_feats = pd.read_csv(save_fname)

        # Filter by SNR and pattern distance thresholds
        all_waveform_feats = filter_by_thresholds(
            all_waveform_feats, snr_threshold=snr_threshold,
            pattern_dist_threshold=pattern_dist_threshold)

        # Take top component for each rhythm type
        all_waveform_feats = all_waveform_feats.loc[all_waveform_feats.groupby(
            ['EID', 'Rhythm'])['comp'].idxmin()].reset_index(drop=True)
        return all_waveform_feats


def t_test_each_feat(
        contrast_feat, feats_df, feats, multicomp='bonf', within=True):
    """Run t-test for each waveform feature.

    Parameters
    ----------
    contrast_feat : str
        Feature to contrast waveform features by.
    feats_df : pd.DataFrame
        DataFrame containing waveform features.
    feats : dict
        Dictionary containing waveform feature names and labels.
    multicomp : str
        Type of multiple comparison correction to use.
    within : bool
        If True, run within-subjects t-test. If False, run between-subjects
        t-test.

    Returns
    -------
    big_df : pd.DataFrame
        DataFrame containing t-test results.
    """
    # Balance data for paired t-test
    if within:
        feats_df_within = feats_df[feats_df.duplicated(['EID'], keep=False)]


    # Initialize DataFrame for t-test results
    big_df = pd.DataFrame([])
    for feat in feats.keys():
        # Perform between-subjects t-test for waveform feature
        ttest = pg.pairwise_tests(
            data=feats_df, dv=feat, between=contrast_feat, effsize='cohen')
        ttest['Feature'] = feat
        big_df = pd.concat((big_df, ttest))

        if within:
            # Perform within subjects t-test for waveform feature
            ttest_within = pg.pairwise_tests(
                data=feats_df_within, dv=feat, within=contrast_feat,
                subject='EID', effsize='cohen')
            ttest_within['Feature'] = feat
            big_df = pd.concat((big_df, ttest_within))

    # Correct for multiple comparison
    _, p_corr = pg.multicomp(big_df['p-unc'].values, method=multicomp)
    big_df['p-corr'], big_df['p-adjust'] = p_corr, multicomp
    return big_df


def plot_each_feat(
        contrast_feat, feats_df, feats, feat_kind, axes=None, stats=None,
        palette=None, order=None, fontsize=11, pairs=None, sig_annot_offset=0.1,
        save_folder=FIG_FOLDER, within=True, markersize=.8, verbose=True):
    """Plot waveform features across desired contrast feature.

    Parameters
    ----------
    contrast_feat : str
        Feature to contrast waveform features by.
    feats_df : pd.DataFrame
        DataFrame containing waveform features.
    feats : dict
        Dictionary containing waveform feature names and labels to plot.
    feat_kind : str
        Description of waveform features for plot title.
    axes : matplotlib.axes.Axes | None
        Axes to plot on. If None, will create new figure.
    stats : pd.DataFrame | None
        DataFrame containing results of t-tests.
    palette : list | None
        Palette to use for violin plots. If None, will use default seaborn
        palette.
    order : list | None
        Order to plot violin plots in. If None, will use default seaborn order.
    fontsize : int
        Fontsize for plot.
    pairs : list | None
        List of pairs to annotate significance for on plot.
    sig_annot_offset : float
        Offset for significance annotation.
    save_folder : str
        Path to folder where figure should be saved.
    within : bool
        If True, plot within-subjects comparisons.
    markersize : float
        Size of markers for swarm plot.
    verbose : bool
        If True, print number of subjects in each plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing violin plots.
    """
    # Make figure for all bycycle features
    if axes is None:
        fig, axes = plt.subplots(
            1 + within, len(feats), figsize=(6 * len(feats), 8 + 8 * within))

    # Iterate over each feature
    for i, (feat, label) in enumerate(feats.items()):
        # Get axis
        try:
            ax = axes.flat[i]
        except AttributeError:
            ax = axes

        # Make violin + swarm plot to show bycycle features by rhythm between
        # subjects
        sns.swarmplot(
            x=contrast_feat, y=feat, data=feats_df, color="white",
            edgecolor="gray", size=markersize, hue="Rhythm", ax=ax, order=order)
        sns.violinplot(
            x=contrast_feat, y=feat, data=feats_df, inner=None, ax=ax,
            palette=palette, order=order, cut=0, split=True)
        ax.set_xlabel(contrast_feat, fontsize=fontsize)
        ax.set_ylabel(label, fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=0.8 * fontsize)

        # Add between-subjects statistics to plot
        if stats is not None and 'Paired' in stats.columns:
            t, d = stats.query('Paired == False')[['T', 'cohen']].iloc[i]
            ax.set_title(
                'T={:.2f}, d={:.3f}'.format(t, d), fontsize=fontsize)

        # Add significance for stats to plot
        if pairs is not None:
            annot = Annotator(
                ax, pairs, data=feats_df, x=contrast_feat,
                y=feat, order=order)
            annot._pvalue_format.fontsize = fontsize
            if stats is not None:
                # Get p-values for each pair if t-test
                if 'Paired' in stats.columns:
                    stats_pairs = stats[stats.set_index(
                        ['A', 'B']).index.isin(pairs)].query('Paired == False')
                    p = list(stats_pairs['p-corr'].iloc[len(pairs)*i:][:len(
                        pairs)])
                # Get p-values for each pair if ANCOVA
                else:
                    p = list(np.repeat(
                        stats.query('Feature == @feat')['p-corr'], len(pairs)))
                annot.configure(test=None, line_width=2.5, verbose=False)
                annot.set_pvalues(pvalues=p)
            else:
                annot.configure(
                    test='t-test_ind', line_width=2.5, verbose=False)
                annot.apply_test()
            annot.annotate(line_offset_to_group=sig_annot_offset)
        sns.despine(ax=ax)

        if within:
            # Get axis
            try:
                ax = axes.flat[i + len(feats)]
            except AttributeError:
                ax = axes

            # Pivot data for within-subjects comparisons
            pivoted = feats_df[['EID', contrast_feat, feat]].pivot(
                index='EID', columns=contrast_feat)
            pivoted['Age'] = feats_df.drop_duplicates(
                subset=['EID'])['Age'].values

            # Make scatterplot to show within subject comparisons
            sns.scatterplot(
                x=(feat, 'Alpha'), y=(feat, 'Mu'), hue='Age', data=pivoted,
                palette='viridis_r', ax=ax, s=15, alpha=0.8)
            min_val = min(feats_df[feat])
            max_val = max(feats_df[feat])
            bot_lim, top_lim = 0.95*min_val, 1.05*max_val
            ax.set_xlim([bot_lim, top_lim])
            ax.set_ylim([bot_lim, top_lim])
            ax.plot([bot_lim, top_lim], [bot_lim, top_lim], 'k--')
            ax.tick_params(axis='both', labelsize=0.8 * fontsize)
            ax.set_xlabel('alpha\n{}'.format(label), fontsize=fontsize)
            ax.set_ylabel('mu\n{}'.format(label), fontsize=fontsize)

            # Add within-subjects statistics to plot
            if stats is not None:
                t, d = stats.query(
                    'Paired == True')[['T', 'cohen']].iloc[i]
                ax.set_title(
                    'T={:.2f}, d={:.3f}'.format(t, d), fontsize=fontsize)
            sns.despine(ax=ax)
    plt.tight_layout()

    # Print number of subjects in each plot
    num_subjects_between = len(set(feats_df['EID']))
    s = (f'\n{feat_kind}\nNumber of subjects included...\nBetween subjects: '
         f'{num_subjects_between}')
    if within:
        num_subjects_within = len(pivoted.dropna().index.unique())
        s += f'\nWithin subjects: {num_subjects_within}'
    if verbose:
        print(s)
    return fig
