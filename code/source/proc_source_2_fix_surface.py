""" The problem adressed by this script: The inner skull intersects with
    the outer_skull surface, making lead field computation impossible.
    Therefore we cut off the lowest part of the inner skull at a
    boundary of -85 and overwrite the surface.
"""
import mne
import numpy as np
from core.params import *


def adjust_surface(
        mri_folder=MRI_FOLDER, subj=PARCELLATION_SUBJ,
        threshold=INNER_SKULL_THRESHOLD):
    """Adjust the inner skull surface by cutting off the lowest part.

    Parameters
    ----------
    mri_folder : str
        Path to MRI folder.
    subj : str
        Subject name.
    threshold : int
        Threshold for cutting off the surface.
    """
    # %% inspect the z-axis of the two surfaces
    outer_skull_fname = f'{mri_folder}/{subj}/bem/outer_skull.surf'
    outer_skull_rr, _ = mne.read_surface(outer_skull_fname)
    print(outer_skull_rr.min(axis=0))

    inner_skull_fname = f'{mri_folder}/{subj}/bem/inner_skull.surf'
    inner_skull_rr, tris = mne.read_surface(inner_skull_fname)
    print(inner_skull_rr.min(axis=0))

    # %% adjust the surface
    idx = np.where(inner_skull_rr[:, 2] < threshold)[0]
    inner_skull_rr[idx, 2] = threshold
    print(inner_skull_rr.min(axis=0))
    mne.write_surface(
        f'{mri_folder}/{subj}/bem/inner_skull.surf', inner_skull_rr, tris,
        overwrite=True)

    # visualize for control
    mne.viz.plot_bem(subj, mri_folder)


if __name__ == '__main__':
    adjust_surface()
