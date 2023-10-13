"""Determines source locations for SSD components using template matching.
"""
# %% Import packages
import mne
import pandas as pd
from scipy.spatial.distance import cdist
from core.helper import get_subject_list, get_source_labels
from core.params import *


def find_closest_area(labels, point, trans_fname=LF_TRANS_FNAME):
    """Find the closest region from the parcellation for a specified point.

    Parameters
    ----------
    labels : list
        List of labels from mne.read_labels_from_annot.
    point : array
        3D coordinates of a point.

    Returns
    -------
    label : mne.Label
        Closest region.
    """
    trans = mne.read_trans(trans_fname)
    point = mne.transforms.apply_trans(trans['trans'], point)
    distances = [np.min(np.sum(np.abs(
        label.pos-point)**2, axis=1)) for label in labels]
    idx = np.argmin(distances)
    return labels[idx]


def compute_pattern_location(df_patterns, sparam_df, fwd, labels):
    """Returns a new DataFrame with xyz-location and classification as mu or
    alpha.

    Parameters
    ----------
    df_patterns : pd.DataFrame
        DataFrame as saved by proc_2 ssd procedure.
    fwd : mne.Forward
        Forward model.
    labels : list
        List of labels.

    Returns
    -------
    df_loc : pd.DataFrame
        DataFrame containing location and binary alpha/mu classification.
    """
    # Reorder according to leadfield
    chs_in_both = list(set(fwd['sol']['row_names']) & set(df_patterns.columns))
    df = df_patterns[fwd.pick_channels(chs_in_both)['sol']['row_names']]
    LF = fwd["sol"]["data"].T
    pos_brain = fwd["source_rr"]

    patterns = df.values.T

    # Compute absolute cosine distance
    distances = cdist(LF, patterns.T, metric="cosine")
    distances = 1 - np.abs(1 - distances)

    # Return xyz-coordinates of minimum distance node
    idx = np.argmin(distances, axis=0)
    mni_pos = pos_brain[idx]

    regions = []
    for point in mni_pos:
        region = find_closest_area(labels, point)
        regions.append(region.name)

    # Create a new DataFrame
    data = (sparam_df['i_comp'].values, mni_pos[:, 0], mni_pos[:, 1],
            mni_pos[:, 2], regions)
    df_loc = pd.DataFrame(data).T
    df_loc.columns = ['idx', 'x', 'y', 'z', 'region']

    # Get alpha and mu regions
    alpha_regions, mu_regions = get_source_labels()

    # If closest node is in this regions, it is classified as mu
    mu_regions = '|'.join(mu_regions)
    mu = df_loc.region.str.contains(mu_regions)
    df_loc.insert(5, 'mu', mu)

    # If closest node is in this regions, it is classified as alpha
    alpha_regions = '|'.join(alpha_regions)
    alpha = df_loc.region.str.contains(alpha_regions)
    df_loc.insert(6, 'alpha', alpha)

    # Add pattern distance
    df_loc.insert(7, 'node_idx', idx)
    df_loc.insert(8, 'pattern_dist', np.min(distances, axis=0))
    return df_loc


def get_sources_one_subj(
        participant, fwd, labels, ssd_sparam_folder=SSD_SPARAM_FOLDER):
    """Determines source locations for SSD components using template matching
    for one subject.

    Parameters
    ----------
    participant : str
        Subject ID.
    fwd : mne.Forward
        Forward model.
    labels : list
        List of labels.
    ssd_sparam_folder : str
        Path to folder with SSD spectral parameters.

    Returns
    -------
    source_df : pd.DataFrame
        DataFrame with source locations.
    """
    # Load data for subject
    patterns_df = pd.read_csv(f'{ssd_sparam_folder}/{participant}_patterns.csv')
    sparam_df = pd.read_csv(f'{ssd_sparam_folder}/{participant}_spec_param.csv')
    source_df = compute_pattern_location(patterns_df, sparam_df, fwd, labels)
    source_df['SNR'] = sparam_df['SNR']
    source_df['subject'] = [participant] * len(source_df.index)

    # Filter out components with no peak detected
    source_df = source_df.loc[sparam_df['PF'].notnull(), :]
    return source_df


def get_sources_all_subjs(
        save_fname=SOURCES_CSV, mri_folder=MRI_FOLDER,
        parcellation_subj=PARCELLATION_SUBJ,
        annotation=PARCELLATION_ANNOTATION):
    """Determines source locations for SSD components using template matching
    for all subjects.

    Parameters
    ----------
    save_fname : str
        Path to save CSV file with source locations.
    mri_folder : str
        Path to MRI folder.
    parcellation_subj : str
        Name of subject with parcellation.
    annotation : str
        Name of annotation.

    Returns
    -------
    df_all : pd.DataFrame
        DataFrame with source locations.
    """
    # See if file already processed
    if os.path.exists(save_fname):
        return pd.read_csv(save_fname)

    # Get subjects
    subjects = get_subject_list('ssd_sparam')

    # load template leadfield
    fwd_fname = f"{mri_folder}/eeg_fwd.fif"
    fwd = mne.read_forward_solution(fwd_fname)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)

    # load patterns
    labels = mne.read_labels_from_annot(
        parcellation_subj, annotation, 'both', subjects_dir=mri_folder,
        verbose=False)

    # Load data
    df_all = []
    for i_sub, participant in enumerate(subjects):
        print(f'Subject {participant} ({i_sub + 1}/{len(subjects)})')
        subj_df = get_sources_one_subj(participant, fwd, labels)
        df_all.append(subj_df)
    df_all = pd.concat(df_all)

    # Save data out to avoid unnecesary re-processing
    df_all.to_csv(save_fname)
    return df_all


if __name__ == '__main__':
    get_sources_all_subjs()