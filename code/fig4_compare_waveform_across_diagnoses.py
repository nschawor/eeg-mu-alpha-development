"""Compare waveform features across diagnoses (Figure 4).
"""
#%% Import necessary modules
import pandas as pd
from itertools import combinations, product
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from core.params import *
from core.helper import load_waveform_features, t_test_each_feat, \
    plot_each_feat


def _merge_waveform_feats_with_diagnoses(
        waveform_feats_df, diagnoses_csv=DIAGNOSES_CSV,
        diagnoses_to_plot=DIAGNOSES_TO_PLOT, map=True):
    """Merge waveform features with diagnostic data.

    Parameters
    ----------
    waveform_feats_df : pd.DataFrame
        DataFrame of waveform features.
    diagnoses_csv : str, optional
        Path to CSV containing diagnostic data, by default DIAGNOSES_CSV.
    diagnoses_to_plot : dict, optional
        Dictionary of diagnoses to plot, by default DIAGNOSES_TO_PLOT.
    map : bool, optional
        Whether to map diagnoses to plot, by default True.

    Returns
    -------
    waveform_feats_df : pd.DataFrame
        DataFrame of waveform features with diagnostic data.
    """
     # Add diagnostic data to bycycle features data
    diagnoses_df = pd.read_csv(diagnoses_csv, index_col=0)
    diagnoses_df = diagnoses_df.rename(columns={
        'subject': 'EID', 'diagnosis': 'Diagnosis'})
    diagnoses_df = diagnoses_df[diagnoses_df['Diagnosis'].isin(
        values=list(diagnoses_to_plot.keys()))]
    if map:
        diagnoses_df['Diagnosis'] = diagnoses_df['Diagnosis'].map(
            diagnoses_to_plot)
    waveform_feats_df = waveform_feats_df.merge(diagnoses_df, on='EID')
    return waveform_feats_df


def _split_waveform_feats_by_rhythm(waveform_feats_df):
    """Split waveform features by rhythm.

    Parameters
    ----------
    waveform_feats_df : pd.DataFrame
        DataFrame of waveform features.

    Returns
    -------
    rhythm_feats : list
        List of DataFrames of waveform features, split by rhythm.
    """
    # Split waveform features by rhythm
    rhythm_feats = [waveform_feats_df.query(
        "Rhythm == @rhythm") for rhythm in ['Alpha', 'Mu']]
    return rhythm_feats


def _t_test_across_diagnoses(
        alpha_feats, mu_feats, feats_to_plot, feat_kind, covar='Age',
        multicomp='fdr_bh', verbose=True):
    """Perform t-test for each waveform feature across diagnoses.

    Parameters
    ----------
    alpha_feats : pd.DataFrame
        DataFrame of alpha waveform features.
    mu_feats : pd.DataFrame
        DataFrame of mu waveform features.
    feats_to_plot : dict
        Dictionary of waveform features to plot.
    feat_kind : str
        Kind of waveform features.
    covar : str, optional
        Covariate to use in t-test, by default 'Age'. If None, no covariate is
        used.
    multicomp : str, optional
        Method for multiple comparisons correction, by default 'fdr_bh'.
    verbose : bool, optional
        Whether to print statistics, by default True.

    Returns
    -------
    stats_alpha : pd.DataFrame
        DataFrame of t-test statistics for alpha features.
    stats_mu : pd.DataFrame
        DataFrame of t-test statistics for mu features.
    """
    # Remove effect of covariate
    if covar is not None:
        for feat in feats_to_plot:
            # Calculate residual for alpha
            alpha_coefs = pg.linear_regression(
                alpha_feats[covar], alpha_feats[feat])['coef'].values
            alpha_feats[feat] = alpha_feats[feat] - (
                alpha_coefs[0] + alpha_coefs[1] * alpha_feats[covar])

            # Calculate residual for mu
            mu_coefs = pg.linear_regression(
                mu_feats[covar], mu_feats[feat])['coef'].values
            mu_feats[feat] = mu_feats[feat] - (
                mu_coefs[0] + mu_coefs[1] * mu_feats[covar])

    # Calculate t-test for each waveform feature
    rhythm_stats = [t_test_each_feat(
        'Diagnosis', feats, feats_to_plot, within=False,
        multicomp=multicomp) for feats in (alpha_feats, mu_feats)]

    # Print t-test results if desired
    if verbose:
        for rhythm, stats in zip(['Alpha', 'Mu'], rhythm_stats):
            print(f'\n{rhythm} {feat_kind}\n{stats}')
    return rhythm_stats


def _ancova_across_diagnoses(alpha_feats, mu_feats, waveform_feats):
    """Perform ANCOVA for each waveform feature across diagnoses.

    Parameters
    ----------
    alpha_feats : pd.DataFrame
        DataFrame of alpha waveform features.
    mu_feats : pd.DataFrame
        DataFrame of mu waveform features.
    waveform_feats : dict
        Dictionary of waveform features to compare.

    Returns
    -------
    stats_alpha : pd.DataFrame
        DataFrame of ANCOVA statistics for alpha features.
    stats_mu : pd.DataFrame
        DataFrame of ANCOVA statistics for mu features.
    """
    # Check if age is a confounding variable for each rhythm
    stats_df = pd.DataFrame([])
    for rhythm, feats_df in zip(['Alpha', 'Mu'], [alpha_feats, mu_feats]):
        for feat in waveform_feats.keys():
            # Perform ANCOVA for feature
            df = pg.ancova(
                data=feats_df, dv=feat, between='Diagnosis', covar='Age')
            pval = df['p-unc'][0]
            F = df['F'][0]
            np2 = df['np2'][0]
            stats_df = pd.concat([stats_df, pd.DataFrame({
                'Rhythm': [rhythm], 'Feature': [feat],
                'F': [F], 'np2': [np2], 'p-unc': [pval]})])

    # Correct for multiple comparisons
    stats_df['p-corr'] = pg.multicomp(
        stats_df['p-unc'].values, method='fdr_bh')[1]

    # Print ANCOVA results
    stats_alpha = stats_df.query('Rhythm == "Alpha"')
    stats_mu = stats_df.query('Rhythm == "Mu"')

    # Print ANCOVA results
    print('ANCOVA - alpha')
    pg.print_table(stats_alpha, floatfmt='.3f')
    print()
    print('ANCOVA - mu')
    pg.print_table(stats_mu, floatfmt='.3f')
    return stats_alpha, stats_mu


def _plot_waveform_feats_across_diagnoses(
        alpha_feats, mu_feats, stats_alpha, stats_mu, feats_to_plot, feat_kind,
        diagnoses_to_plot=DIAGNOSES_TO_PLOT,
        diagnoses_palette=DIAGNOSES_PALETTE, verbose=True):
    """Plot waveform features across diagnoses.

    Parameters
    ----------
    alpha_feats : pd.DataFrame
        DataFrame of alpha waveform features.
    mu_feats : pd.DataFrame
        DataFrame of mu waveform features.
    stats_alpha : pd.DataFrame
        DataFrame of statistics for alpha features.
    stats_mu : pd.DataFrame
        DataFrame of statistics for mu features.
    feats_to_plot : dict
        Dictionary of waveform features to plot.
    feat_kind : str
        Kind of waveform features.
    diagnoses_to_plot : dict, optional
        Dictionary of diagnoses to plot, by default DIAGNOSES_TO_PLOT.
    diagnoses_palette : dict, optional
        Dictionary of palettes for each rhythm, by default DIAGNOSES_PALETTE.
    verbose : bool, optional
        Whether to print statistics, by default True.

    Returns
    -------
    figs : list
        List of matplotlib Figures.
    """
    # Plot waveform features across diagnoses
    diagnoses = sorted(list(set(diagnoses_to_plot.values())))
    pairs = [pair for pair in combinations(
        diagnoses, 2) if diagnoses_to_plot['No Diagnosis Given'] in pair]
    figs = []
    for feats, rhythm, stats in zip(
            [alpha_feats, mu_feats], ['Alpha', 'Mu'], [stats_alpha, stats_mu]):
        fig = plot_each_feat(
            'Diagnosis', feats, feats_to_plot, f'{rhythm} {feat_kind}',
            palette=diagnoses_palette[rhythm], order=diagnoses, pairs=pairs,
            stats=stats, sig_annot_offset=0.07, within=False, markersize=1.5,
            verbose=verbose)
        fig.set_size_inches(8, 3)
        for ax in fig.axes:
            ax.legend_ = None
            ax.set_title('')
            ax.set_xlabel('diagnosis')
        fig.subplots_adjust(wspace=0.4)
        plt.style.use('figures.mplstyle')
        save_fname = f'{FIG_FOLDER}/fig4_diagnosis_{rhythm}.pdf'
        plt.savefig(save_fname, dpi=300, bbox_inches='tight')
        figs.append(fig)
    return figs


def waveform_feats_across_diagnoses(
        alpha_feats, mu_feats, feats_to_plot, feat_kind, stat='t-test',
        verbose=True):
    """Perform statistics and plot waveform features across diagnoses.

    Parameters
    ----------
    alpha_feats : pd.DataFrame
        DataFrame of alpha waveform features.
    mu_feats : pd.DataFrame
        DataFrame of mu waveform features.
    feats_to_plot : dict
        Dictionary of waveform features to plot.
    feat_kind : str
        Kind of waveform features.
    stat : str, optional
        Statistical test to perform, by default 'ancova'. Options are 't-test'
        and 'ancova'.
    verbose : bool, optional
        Whether to print statistics, by default True.

    Returns
    -------
    stats_alpha : pd.DataFrame
        DataFrame of statistics for alpha features.
    stats_mu : pd.DataFrame
        DataFrame of statistics for mu features.
    fig : matplotlib Figure
        Figure containing plot.
    """

    # T-test across diagnoses
    ttest_alpha, ttest_mu = _t_test_across_diagnoses(
        alpha_feats, mu_feats, feats_to_plot, feat_kind, verbose=verbose)

    # ANCOVA across diagnoses
    ancova_alpha, ancova_mu = _ancova_across_diagnoses(
        alpha_feats, mu_feats, feats_to_plot)

    # Choose statistics to plot
    if stat == 't-test':
        stats_alpha, stats_mu = ttest_alpha, ttest_mu
    elif stat == 'ancova':
        stats_alpha, stats_mu = ancova_alpha, ancova_mu

    # Plot waveform features across diagnoses
    fig = _plot_waveform_feats_across_diagnoses(
        alpha_feats, mu_feats, stats_alpha, stats_mu, feats_to_plot, feat_kind,
        verbose=verbose)
    return stats_alpha, stats_mu, fig


def sensitivity_analysis(
        feats_to_plot, feat_kind, snr_thresholds=SNR_THRESHOLDS,
        pattern_dist_thresholds=PATTERN_DIST_THRESHOLDS,
        palettes=DIAGNOSES_PALETTE, save_folder=FIG_FOLDER):
    """Perform sensitivity analysis on waveform features for each combination
    of SNR and pattern distance thresholds.

    Parameters
    ----------
    feats_to_plot : dict
        Dictionary of waveform features to plot.
    feat_kind : str
        Kind of waveform features.
    snr_thresholds : list, optional
        List of SNR thresholds to use, by default SNR_THRESHOLDS.
    pattern_dist_thresholds : list, optional
        List of pattern distance thresholds to use, by default
        PATTERN_DIST_THRESHOLDS.
    palettes : dict, optional
        Dictionary of palettes for each rhythm, by default DIAGNOSES_PALETTE.
    save_folder : str, optional
        Folder to save figures to, by default FIG_FOLDER.

    Returns
    -------
    stats_df : pd.DataFrame
        DataFrame containing statistics for each combination of thresholds.
    fig : matplotlib Figure
        Figure containing plot.
    """
    # Create DataFrame to store statistics
    stats_df = pd.DataFrame([])

    # Iterate through thresholds
    for snr_threshold, pattern_dist_threshold in product(
            snr_thresholds, pattern_dist_thresholds):
        # Load waveform features
        waveform_feats = load_waveform_features(
            snr_threshold=snr_threshold,
            pattern_dist_threshold=pattern_dist_threshold)

        # Merge waveform features with diagnostic data
        waveform_feats = _merge_waveform_feats_with_diagnoses(waveform_feats)

        # Split waveform features by rhythm
        rhythm_feats = _split_waveform_feats_by_rhythm(waveform_feats)

        # Calculate t-tests
        rhythm_stats = _t_test_across_diagnoses(
            *rhythm_feats, feats_to_plot, feat_kind, verbose=False)
        print(rhythm_stats)

        # Include thresholds in DataFrame
        for rhythm, feats, stats, in zip(
                ['Alpha', 'Mu'], rhythm_feats, rhythm_stats):
            stats[['SNR', 'Pattern Distance', 'Rhythm', 'n']] = (
                snr_threshold, pattern_dist_threshold, rhythm,
                len(set(feats['EID'])))
            stats_df = pd.concat([stats_df, stats.query('B == "None"').rename(
                columns={'A': 'Diagnosis'})[
                    ['Rhythm', 'Diagnosis', 'Feature', 'p-corr', 'SNR',
                    'Pattern Distance', 'n']]])

    # Make plot for sensitivity analysis
    colors = zip(palettes['Alpha'], palettes['Mu'])
    for diagnosis, color_set in zip(stats_df['Diagnosis'].unique(), colors):
        # Make figure
        fig, axes = plt.subplots(2, len(feats_to_plot), figsize=(
            8, 4))

        # Seperate stats by diagnosis
        stats_df_one_diagnosis = stats_df.query('Diagnosis == @diagnosis')

        # Make colormaps for sensitivity analysis plot
        for i, feat in enumerate(feats_to_plot):
            for j, (rhythm, c) in enumerate(zip(['Alpha', 'Mu'], color_set)):
                # Seperate alpha and mu features
                stats = stats_df_one_diagnosis.query('Rhythm == @rhythm').round(
                    decimals=3)

                # Plot heatmap of p-values
                pvals = stats.query('Feature == @feat').pivot(
                    index='SNR', columns='Pattern Distance')['p-corr']
                cmap = sns.blend_palette([c, 'gray', 'white'], as_cmap=True)
                sns.heatmap(
                    pvals, ax=axes[j, i], vmin=0, vmax=1, annot=True,
                    fmt='.3f', cmap=cmap, linecolor='k', linewidths=1)

            # Set title
            axes[0, i].set_title(feats_to_plot[feat], fontsize=20)
            plt.suptitle(f'Sensitivity Analysis - {diagnosis}', fontsize=30)

        # Save figure
        feat_kind_fname = '_'.join(feat_kind.lower().split(' '))
        save_fname = (f'{save_folder}/{feat_kind_fname}_sensitivity_'
                      f'{diagnosis}.pdf')
        plt.savefig(save_fname, dpi=300, bbox_inches='tight', transparent=True)
    return stats_df, fig


def check_adhd_subtypes(
        waveform_feats, diagnoses_to_plot, waveform_feats_to_plot):
    """Check if ADHD subtypes are different from each other.

    Parameters
    ----------
    waveform_feats : pd.DataFrame
        DataFrame of waveform features.
    diagnoses_to_plot : dict
        Dictionary of diagnoses to plot.
    waveform_feats_to_plot : dict
        Dictionary of waveform features to plot.
    """
    # Merge waveform features with diagnostic data
    feats_by_diagnoses = _merge_waveform_feats_with_diagnoses(
        waveform_feats, map=False)

    # Remove non-ADHD diagnoses
    adhd_subtypes = [k for k, v in diagnoses_to_plot.items() if 'ADHD' in v]
    feats_by_diagnoses = feats_by_diagnoses.query('Diagnosis in @adhd_subtypes')

    # Split waveform features by rhythm
    alpha_feats, mu_feats = _split_waveform_feats_by_rhythm(feats_by_diagnoses)

    # Run t-tests
    stats_alpha, stats_mu = _t_test_across_diagnoses(
        alpha_feats, mu_feats, waveform_feats_to_plot, 'Waveform Features',
        verbose=False)

    # Print results
    for rhythm, stats in zip(['Alpha', 'Mu'], [stats_alpha, stats_mu]):
        print(f'\nADHD Subtypes - {rhythm}\n{stats}')


if __name__ == '__main__':
    # Get waveform features
    waveform_feats = load_waveform_features()

    # Merge waveform features with diagnostic data
    feats_by_diagnoses = _merge_waveform_feats_with_diagnoses(waveform_feats)
    print("Alpha")
    print(feats_by_diagnoses[feats_by_diagnoses.Rhythm=="Alpha"].value_counts(
        'Diagnosis'))
    print("MU")
    print(feats_by_diagnoses[feats_by_diagnoses.Rhythm=="Mu"].value_counts(
        'Diagnosis'))

    # Split waveform features by rhythm
    alpha_feats, mu_feats = _split_waveform_feats_by_rhythm(feats_by_diagnoses)

    # Plot waveform features across diagnoses
    stats_alpha, stats_mu, fig = waveform_feats_across_diagnoses(
        alpha_feats, mu_feats, WAVEFORM_FEATS_TO_PLOT, 'Waveform Features')
# %%
