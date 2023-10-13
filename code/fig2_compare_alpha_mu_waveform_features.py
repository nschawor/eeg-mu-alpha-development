"""Compare waveform features between alpha and mu rhythms (Figure 2C-D).
"""
#%% Import necessary modules
import pingouin as pg
import pandas as pd
from core.params import *
from core.helper import load_waveform_features, t_test_each_feat, \
    get_alpha_mu_comps
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator


def compare_num_generators(all_subjects_csv=ALL_SUBJECTS_CSV):
    """Compare number of alpha and mu generators between subjects.

    Parameters
    ----------
    all_subjects_csv : str
        Path to CSV containing all subjects.

    Returns
    -------
    num_generators_df : pandas.DataFrame
        DataFrame containing number of alpha and mu generators for each subject.
    """
    # Get alpha and mu components
    alpha_comps, mu_comps = get_alpha_mu_comps()

    # Count number of alpha and mu generators for each subject
    alpha_gen_counts = alpha_comps.value_counts('subject')
    mu_gen_counts = mu_comps.value_counts('subject')

    # Merge alpha and mu generator counts into one DataFrame
    num_generators_df = pd.concat([alpha_gen_counts, mu_gen_counts], axis=1)
    num_generators_df.columns = ['alpha', 'mu']

    # Rename subject column
    num_generators_df = num_generators_df.reset_index().rename(
        columns={'subject': 'EID'})

    # Load ages from CSV
    ages = pd.read_csv(all_subjects_csv, index_col='EID')['Age']

    # Combine age and peak trough symmetry into one DataFrame
    num_generators_df['Age'] = num_generators_df['EID'].map(ages)

    # Impute missing data
    num_generators_df = num_generators_df.fillna(0)

    # Melt DataFrame for plotting
    num_generators_df = num_generators_df.melt(
        id_vars=['EID', 'Age'], value_vars=['alpha', 'mu'], var_name='Rhythm',
        value_name='num_generators')
    return num_generators_df


if __name__ == '__main__':
    # Plot waveform features
    waveform_feats = load_waveform_features()
    waveform_feats.Rhythm.replace({'Alpha': 'alpha', 'Mu': 'mu'}, inplace=True)

    # Initialize figure
    plt.style.use('figures.mplstyle')
    fig, ax = plt.subplots(2, 4, gridspec_kw={'height_ratios': [1.5, 1], 
                                              'left': 0.1, 
                                              'right': 0.98,
                                              'hspace': 0.3, 
                                              'wspace': 0.4})
    fig.set_size_inches(7, 6)

    # Calculate t-test for each waveform feature between subjects
    t_stats_between = t_test_each_feat(
        'Rhythm', waveform_feats, WAVEFORM_FEATS_TO_PLOT)
    print('unpaired')
    pg.print_table(t_stats_between[t_stats_between.Paired==False][['Feature','T', 'cohen', 'p-corr']], floatfmt='.3')
    
    print('paired')
    pg.print_table(t_stats_between[t_stats_between.Paired==True][['Feature','T', 'cohen', 'p-corr']], floatfmt='.3')

    # Iterate over each feature
    for i, (feat, label) in enumerate(WAVEFORM_FEATS_TO_PLOT.items()):
        # Get axis
        ax2 = ax.flat[i]

        # Make violin + swarm plot to show bycycle features by rhythm between
        # subjects
        melted_df = pd.melt(waveform_feats, id_vars=['Rhythm', 'EID'])
        melted_df = melted_df[melted_df.variable==feat]
        fig2, ax1 = plt.subplots()
        a = sns.swarmplot(
            data=waveform_feats, x='Rhythm', y=feat, color='k', size=3, alpha=0.5, ax=ax1)
        fig2.show()
        plt.close(fig2)
        inner_kws=dict(color="w") #, lw=1)
        parts = sns.violinplot(
            x="variable", y='value', data=melted_df, split=True, inner="quart",
            inner_kws=dict(color=".8", linewidth=1, alpha=0.5),
            ax=ax2, hue='Rhythm', palette=(MU_COLOR, ALPHA_COLOR), cut=0)
        XY = a.get_children()[0].get_offsets()
        print(XY.shape)
        ax2.plot(XY[:,0]-.5, XY[:,1], 'k.', mec='none', markersize=1.8, alpha=.8, mfc='k')
        XY = a.get_children()[1].get_offsets()
        print(XY.shape)
        ax2.plot(XY[:,0]-.5, XY[:,1], '.', mec='none', markersize=1.8, alpha=.8, mfc='k')
        ax2.set(xlim=(-.8,1))
        ax2.set_xlabel('')
        ax2.set_ylabel(label)
        ax2.set_xticks([])
        # ax2.set_title(label)
        ax2.tick_params(axis='both')
        ax2.set_ylim(ax2.get_ylim())

        # Add significance for stats to plot
        waveform_feats['bla'] = waveform_feats.Rhythm.replace(
            {'alpha': -4, 'mu': -4})
        # annot = Annotator(
        #     ax2, [('alpha', 'mu')], data=waveform_feats, x='Rhythm',
        #     y=feat, order=None)
        # p = t_stats_between.query('Paired == False')['p-corr'].iloc[i]
        # annot.configure(test=None, line_width=1.5, verbose=False)
        # annot.set_pvalues(pvalues=[p])
        # sig_annot_offset=0.1
        # annot.annotate(line_offset=-1, line_offset_to_group=sig_annot_offset)
        sns.despine(ax=ax2)


    for i, (feat, label) in enumerate(WAVEFORM_FEATS_TO_PLOT.items()):

        # Pivot data for within-subjects comparisons
        pivoted = waveform_feats[['EID', 'Rhythm', feat]].pivot(
            index='EID', columns='Rhythm')
        pivoted['Age'] = waveform_feats.drop_duplicates(
            subset=['EID'])['Age'].values

        # Make scatterplot to show within subject comparisons
        ax3 = ax.flat[i + len(WAVEFORM_FEATS_TO_PLOT)]
        sns.scatterplot(
            x=(feat, 'alpha'), y=(feat, 'mu'), hue='Age', data=pivoted,
            palette='viridis_r', ax=ax3, s=10, alpha=0.7)
        min_val = min(waveform_feats[feat])
        max_val = max(waveform_feats[feat])
        bot_lim, top_lim = 0.99*min_val, 1.01*max_val
        ax3.set_xlim([bot_lim, top_lim])
        ax3.set_ylim([bot_lim, top_lim])
        ax3.plot([bot_lim, top_lim], [bot_lim, top_lim], linestyle='-', color='k', lw=0.5)
        ax3.tick_params(axis='both', labelsize=8)
        ax3.set_yticklabels([])
        ax3.set_xlabel('alpha\n{}'.format(label), fontsize=10)
        ax3.set_ylabel('mu\n{}'.format(label), fontsize=10)
        ax3.set_aspect(1)
    sns.despine(fig=fig)

    # Fix axes aesthetics
    for i in range(4):
        fig.axes[i].set_xticks([-.35, .35])
        fig.axes[i].set_xticklabels(['mu', 'alpha'])
        fig.axes[i].set_xlabel('')
        # fig.axes[i].get_children()[0].set_edgecolor(None)
        # fig.axes[i].get_children()[1].set_edgecolor(None)

    # Turn off legend for all but one axis
    for i in range(4):
        fig.axes[i].legend('', frameon=False)
    # fig.axes[0].legend(title='rhythm', frameon=True, bbox_to_anchor=[0.1, -0.05])
    # Add legend for age
    # leg = fig.axes[4].legend(title='age', bbox_to_anchor=[1, 2], ncol=1)
    for i in range(4, 8):
        fig.axes[i].legend([], frameon=False)
    sns.despine(fig)
    fig.show()

    # Save figure
    os.makedirs(FIG_FOLDER, exist_ok=True)
    fig.savefig(f'{FIG_FOLDER}/fig2_mu_vs_alpha.pdf', dpi=200, transparent=True)
    fig.show()

# %%
