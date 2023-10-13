"""Plot histogram of sources by region on 3D brain (Figure 1E).
"""
# %% Import necessary modules
import numpy as np
import mne
import pandas as pd
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from core import params
from core.helper import get_source_labels
# matplotlib.use('MacOSX')


def plot_brain_histogram(
        sources_df, hemisphere, snr_threshold=0.0, dist_threshold=1,
        views=('lateral', 'medial'), mri_folder=params.MRI_FOLDER,
        cbar_lims=(params.CBAR_MIN, params.CBAR_MID, params.CBAR_MAX),
        parcellation_subj=params.PARCELLATION_SUBJ,
        annotation=params.PARCELLATION_ANNOTATION,
        save_folder=params.FIG_FOLDER):
    """Plot histogram of sources by region on 3D brain.

    Parameters
    ----------
    sources_df : pandas.DataFrame
        DataFrame containing source localization data.
    hemisphere : str
        Hemisphere to plot ('lh', 'rh', or 'both').
    snr_threshold : float
        SNR threshold to use. Components with SNR below this threshold will be
        excluded from the plot.
    dist_threshold : float
        Distance threshold to use. Components with distance above this
        threshold will be excluded from the plot.
    views : tuple of str
        Views to plot ('lateral', 'medial', 'rostral', 'caudal', 'dorsal',
        'ventral').
    mri_folder : str
        Folder containing MRI data.
    cbar_lims : tuple of float
        Color bar limits (min, mid, max).
    parcellation_subj : str
        Subject to use for parcellation.
    annotation : str
        Parcellation to use.
    save_folder : str
        Folder to save figure in.

    Returns
    -------
    img : numpy.ndarray
        Image of brain histogram.
    cbar_img : numpy.ndarray
        Image of color bar.
    """
    # Parse hemisphere input
    if hemisphere == 'both':
        hemispheres = ('lh', 'rh')
    else:
        hemispheres = (hemisphere,)

    # Parse views input
    if type(views) == str:
        views = (views,)

    # Threshold by SNR and distance
    sources_df = sources_df.query(f'SNR > {snr_threshold}')
    sources_df = sources_df.query(f'`pattern_dist` < {dist_threshold}')

    # Make colormap
    cmap = sns.blend_palette(
        ['grey', 'grey', 'maroon', 'maroon', 'firebrick', 'red', 'orangered',
         'orange', 'gold'], as_cmap=True)

    # Combine all hemispheres and views into one image
    imgs = []
    for hemi in hemispheres:
        # Get labels for annotation
        labels = mne.read_labels_from_annot(
            parcellation_subj, annotation, hemi, subjects_dir=mri_folder,
            verbose=False)

        # Map percentages to colors
        sources_df = sources_df[~sources_df['region'].str.contains(
            '?', regex=False)]
        region_counts = sources_df['region'].value_counts()
        region_pcts = region_counts / region_counts.sum()
        regions = list(region_counts.to_dict().keys())
        alpha_regions, mu_regions = get_source_labels(names=False)

        # Extract labels, data and vertices to plot
        labels_to_plot = [l for l in labels if l.name in regions]
        data, vertices = [], []
        for l in labels_to_plot:
            data.extend([region_pcts.to_dict()[l.name]] * len(l.vertices))
            vertices.extend(l.vertices)
        data, vertices = np.array(data), np.array(vertices)

        # Plot brain with data by region
        brain = mne.viz.Brain(
            parcellation_subj, hemi, 'inflated', subjects_dir=mri_folder,
            cortex='low_contrast', background='white', size=(800, 600))
        brain.add_annotation(annotation, color='w', alpha=.6)
        fmin, fmid, fmax = cbar_lims
        brain.add_data(
            data, vertices=vertices, fmin=fmin, fmid=fmid, fmax=fmax,
            transparent=True, colorbar=False, colormap=cmap)

        # Outline alpha and mu regions
        for region in alpha_regions:
            if region.name.startswith(hemi[0].capitalize()):
                brain.add_label(region, borders=True, color=params.ALPHA_COLOR)
        for region in mu_regions:
            if region.name.startswith(hemi[0].capitalize()):
                brain.add_label(region, borders=True, color=params.MU_COLOR)

        # Show different views
        img_views = []
        for view in views:
            # Rotate to desired view
            brain.show_view(view)

            # Take screenshot
            img = brain.screenshot()

            # Resize if sizing is inconsistent
            if img.shape != (600, 800, 3):
                img = resize(img, (600, 800, 3))
            img_views.append(img[100:500, 100:700, :])
        imgs.append(np.vstack(img_views))
        brain.close()
    img = np.hstack(imgs)

    # Make color bar image
    brain = mne.viz.Brain(
        parcellation_subj, 'lh', 'inflated', subjects_dir=mri_folder,
        cortex='low_contrast', background='white', alpha=0)
    cbar_kwargs = {
        'vertical': True, 'font_family': 'Arial', 'height': 0.8, 'width': 0.15,
        'bold': False, 'label_font_size': 18, 'n_labels': 2, 'fmt': '%.2f'}
    brain.add_data(
        np.zeros(vertices.shape), vertices=vertices, fmin=fmin, fmid=fmid,
        fmax=fmax, transparent=True, colorbar=True, colorbar_kwargs=cbar_kwargs,
        colormap=cmap)
    cbar_img = brain.screenshot()[100:, 100:250, :]
    brain.close()

    # Plot if desired
    if save_folder:
        # Put thresholds in file name
        save_fname = (f'{save_folder}/brain_hist_snr{snr_threshold}'
                      f'_td{dist_threshold}.pdf')

        # Initialize figure
        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(
            nrows=2*len(views), ncols=(2*len(hemispheres))+1)

        # Add brain histograms to figure
        brain_ax = fig.add_subplot(gs[:, 1:])
        brain_ax.imshow(img)
        brain_ax.set_axis_off()

        # Add colorbar for all histograms
        cbar_ax = fig.add_subplot(gs[:, 0])
        cbar_ax.imshow(cbar_img)
        cbar_ax.set_axis_off()

        plt.savefig(save_fname)
    return img, cbar_img


def plot_brain_histogram_2d(
        sources_df, hemisphere, snr_thresholds=params.SNR_THRESHOLDS,
        dist_thresholds=params.PATTERN_DIST_THRESHOLDS,
        save_folder=params.FIG_FOLDER):
    """Plot brain histograms for all combinations of SNR and distance
    thresholds.

    Parameters
    ----------
    sources_df : pandas.DataFrame
        DataFrame containing source localization data.
    hemisphere : str
        Hemisphere to plot ('lh', 'rh', or 'both').
    snr_thresholds : list of float
        SNR thresholds to plot.
    dist_thresholds : list of float
        Distance thresholds to plot.
    save_folder : str
        Folder to save figure in.
    """
    # Initialize figure
    fig = plt.figure(figsize=(
        20 * len(snr_thresholds), 20 * len(dist_thresholds)))
    gs = gridspec.GridSpec(nrows=len(snr_thresholds), ncols=(
        len(dist_thresholds)) + 2)

    # Plot histograms for each combination of SNR and distance thresholds
    for i, snr_threshold in enumerate(snr_thresholds):
        # Label rows
        ax = fig.add_subplot(gs[i, 0])
        ax.text(
            1.05, 0.5, f'SNR > {snr_threshold} dB', size=55,
            weight='bold', ha='right', family='Arial')
        ax.set_axis_off()
        for j, dist_threshold in enumerate(dist_thresholds):
            # Plot histogram for SNR and distance threshold
            img, cbar_img = plot_brain_histogram(
                sources_df, hemisphere, snr_threshold=snr_threshold,
                dist_threshold=dist_threshold, save_folder=None)

            # Plot image on desired axis
            ax = fig.add_subplot(gs[i, j + 1])
            ax.imshow(img)
            ax.set_axis_off()

            # Label columns
            if i == 0:
                ax.set_title(
                    f'Distance < {dist_threshold:.2f}', size=55, pad=100,
                    weight='bold', family='Arial')

    # Plot colorbar
    cbar_ax = fig.add_subplot(gs[1:-1, -1])
    cbar_ax.imshow(cbar_img)
    cbar_ax.set_axis_off()

    # Save figure
    save_fname = f'{save_folder}/brain_hist_2d.pdf'
    plt.savefig(save_fname, dpi=200)


if __name__ == '__main__':
    df_all = pd.read_csv(params.SOURCES_CSV)
    plot_brain_histogram(
        df_all, 'both', snr_threshold=params.SNR_THRESHOLD,
        dist_threshold=params.PATTERN_DIST_THRESHOLD)
# %%
