"""Plot waveform features across age (Figure 3).
"""
# %% Import necessary modules
import seaborn as sns
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from core.helper import load_waveform_features
from core.params import *


def compare_slope_alpha_vs_mu_across_age(
        waveform_feats_df, waveform_feats, method='bootstrap', n_boots=10000,
        age_min=AGE_MIN):
    """Compare slope of waveform features across age for alpha vs mu.

    Parameters
    ----------
    waveform_feats_df : pandas DataFrame
        DataFrame containing waveform features across all subjects.
    waveform_feats : dict
        Dictionary of waveform features to compare.
    method : str, optional
        Method to use for calculating statistics. The default is 'bootstrap'.
    n_boots : int, optional
        Number of bootstrap iterations. The default is 10000.
    age_min : int, optional
        Minimum age to plot. The default is AGE_MIN.

    Returns
    -------
    lr_df : pandas DataFrame
        DataFrame containing slope and intercept for each waveform feature.
    """
    # Ensure method entered is implemented
    assert method in ['bootstrap', 'ancova']

    # Calculate ANCOVA if desired
    if method == 'ancova':
        stats_df = pd.DataFrame([])
        for feat in waveform_feats:
            # Calculate ANCOVA for feature
            stats_one_feat = pg.ancova(
                data=waveform_feats_df, dv=feat, between='Rhythm', covar='Age')
            stats_one_feat['feat'] = feat
            stats_df = pd.concat((stats_df, stats_one_feat))

        # Correct for multiple comparisons
        stats_df['p-corr'] = pg.multicomp(
            stats_df['p-unc'].values, method='bonf')[1]
        print(stats_df)
        return None

    # Split by rhythm
    alpha_feats_df = waveform_feats_df.query("Rhythm == 'alpha'")
    mu_feats_df = waveform_feats_df.query("Rhythm == 'mu'")

    # Calculate slope for each waveform feature and put into one big DataFrame
    lr_fname = '../results/bootstrap.csv'
    if os.path.exists(lr_fname):
        lr_df = pd.read_csv(lr_fname)
    else:
        lr_df, count = pd.DataFrame([]), 0
        for feats_df in (alpha_feats_df, mu_feats_df):
            for boot_num in range(n_boots):
                # Resample with replacement for bootstrapping
                boot_feats_df = feats_df.sample(frac=1, replace=True)
                for feat in waveform_feats:
                    one_lr_df = pg.linear_regression(
                        X=boot_feats_df['Age'] - age_min, y=boot_feats_df[feat])
                    lr_df = pd.concat((lr_df, pd.DataFrame(
                        data={'boot_num': boot_num,
                              'Rhythm': boot_feats_df['Rhythm'].iloc[0],
                              'feat': feat, 'slope': one_lr_df.iloc[1, 1],
                              'intercept': one_lr_df.iloc[0, 1],
                              'r2': one_lr_df.iloc[1, 5]}, index=[count])))
                    count += 1
        lr_df = lr_df.reset_index(drop=True)
    lr_df.to_csv(lr_fname, index=False)

    # Calculate 95% CI and p-value from difference of bootstrapped slopes for
    # each waveform feature
    stats_df, count = pd.DataFrame([]), 0
    for feat in waveform_feats:
        feat_df = lr_df.query("feat == @feat")
        for lr_param in ['slope', 'intercept']:
            param_arr = [v[lr_param].values for _, v in feat_df.groupby(
                'Rhythm')]
            param_diff = np.subtract(*param_arr)
            ci = np.round(np.percentile(param_diff, [2.5, 97.5]), 4)
            p_val = min(sum(param_diff > 0), sum(param_diff < 0)) / n_boots * 2
            stats = pd.DataFrame(
                {'feat': feat, 'lr_param': lr_param, 'p': p_val, 'ci': [ci]},
                index=[count])
            count += 1
            stats_df = pd.concat((stats_df, stats))
    stats_df = stats_df.reset_index(drop=True)

    # Print stats
    print("slope")
    print(stats_df[stats_df.lr_param=="slope"])
    print("\nintercept")
    print(stats_df[stats_df.lr_param=="intercept"])
    return lr_df


def plot_waveform_feats_across_age(
        waveform_feats_df, waveform_feats, feat_kind, method='bootstrap',
        palette=(ALPHA_COLOR, MU_COLOR), age_min=AGE_MIN, age_max=AGE_MAX):
    """Plot waveform features across age.

    Parameters
    ----------
    waveform_feats_df : pandas DataFrame
        DataFrame containing waveform features across all subjects.
    waveform_feats : dict
        Dictionary of waveform features to plot.
    feat_kind : str
        Kind of waveform feature being plotted.
    method : str, optional
        Method to use for calculating statistics. The default is 'bootstrap'.
    palette : tuple, optional
        Colors to use for plotting. The default is (ALPHA_COLOR, MU_COLOR).
    age_min : int, optional
        Minimum age to plot. The default is AGE_MIN.
    age_max : int, optional
        Maximum age to plot. The default is AGE_MAX.

    Returns
    -------
    fig : matplotlib Figure
        Figure containing plot.
    """
    # Ensure method entered is implemented
    assert method in ['bootstrap', 'ancova']

    # Cap age
    waveform_feats_df = waveform_feats_df.query("Age < @age_max")

    # # Compare slope across age for alpha vs mu
    print(f'\n{feat_kind}')
    lr_df = compare_slope_alpha_vs_mu_across_age(
        waveform_feats_df, waveform_feats, method=method)

    # Make figure for all waveform features
    plt.style.use('figures.mplstyle')

    # Initialize figure
    boot_plot = lr_df is not None
    fig = plt.figure(figsize=(7, 3.5 + 2 * boot_plot))
    gs = GridSpec(
        nrows=1 + 2 * boot_plot, ncols=len(waveform_feats), figure=fig,
        hspace=0.6, wspace=0.6,
        left=0.1, right=0.98, top=0.98,
        height_ratios=[3.5, 1, 1] if boot_plot else [1])
    for i, (feat, label) in enumerate(waveform_feats.items()):
        # Get axis for top row
        ax = fig.add_subplot(gs[0, i])

        # Plot waveform features across age
        for i_rhythm, rhythm in enumerate(['alpha', 'mu']):
            R = waveform_feats_df[waveform_feats_df['Rhythm']==rhythm]
            sns.regplot(
                x='Age', y=feat, data=R, scatter=False, color=palette[i_rhythm],
                label=rhythm.lower())
        sns.scatterplot(
            x='Age', y=feat, hue='Rhythm', data=waveform_feats_df, ax=ax,
            palette=palette, alpha=0.4, s=5, hue_order=('alpha', 'mu'),
            legend=False, linewidth=0)
        # if i == 0:
        #     # ax.legend(
        #     #     loc='upper left', bbox_to_anchor=(0.05, 0.95), title='rhythm')
        # else:
        ax.legend_ = None
        ax.tick_params(axis='both')
        ax.set_xlabel('age')
        ax.set_ylabel(label)
        ax.set_xlim(age_min, age_max + 0.3)
        sns.despine(ax=ax)

        # Get axes for bootstrap plots
        if boot_plot:
            ax_middle = fig.add_subplot(gs[1, i])
            ax_bottom = fig.add_subplot(gs[2, i])

            # Plot bootstrap distributions
            sns.boxplot(data=lr_df.query(
                "feat == @feat"), x='slope', y='Rhythm', orient='h',
                ax=ax_middle, palette=palette, hue_order=('alpha', 'mu'),
                showfliers=False, whis=(2.5, 97.5))
            sns.boxplot(data=lr_df.query(
                "feat == @feat"), x='intercept', y='Rhythm', orient='h',
                ax=ax_bottom, palette=palette, hue_order=('alpha', 'mu'),
                showfliers=False, whis=(2.5, 97.5))

            # Plot aesthetics
            ax_middle.set_xlabel('slope (' + r'$\beta_1$' + ')')
            ax_bottom.set_xlabel('intercept (' + r'$\beta_0$' + ')')
            ax_middle.set_ylabel('')
            ax_bottom.set_ylabel('')
            sns.despine(ax=ax_middle)
            sns.despine(ax=ax_bottom)

    # Calculate correlations between age and waveform features
    print(f'\n{feat_kind} Correlations')
    for rhythm in sorted(waveform_feats_df['Rhythm'].unique()):
        rhythm_df = waveform_feats_df.query(f'Rhythm == "{rhythm}"')
        rhythm_df = rhythm_df[['Age'] + list(waveform_feats.keys())]
        stats = pg.pairwise_corr(
            data=rhythm_df, columns=['Age'], method='spearman', padjust='bonf')
        print(f'\n{rhythm.capitalize()}\n{stats}')

    # Save figure
    feat_kind_fname = '_'.join(feat_kind.lower().split(' '))
    os.makedirs(FIG_FOLDER, exist_ok=True)
    save_fname = f'{FIG_FOLDER}/fig3_{feat_kind_fname}_across_age.pdf'
    fig.savefig(save_fname, dpi=300)
    fig.show()
    return fig


if __name__ == '__main__':
    # Plot bycycle features across age
    waveform_feats = load_waveform_features()
    waveform_feats.Rhythm.replace({'Alpha': 'alpha', 'Mu': 'mu'}, inplace=True)

    fig = plot_waveform_feats_across_age(
        waveform_feats, WAVEFORM_FEATS_TO_PLOT, 'Waveform Features')
# %%
