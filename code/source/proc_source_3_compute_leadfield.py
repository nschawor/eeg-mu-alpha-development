""" Compute source model and leadfield for the child template brain."""
# %% Import necessary modules
import os
import mne
from core.params import *


def coregistration(
        fname_electrodes, subj=PARCELLATION_SUBJ, mri_folder=MRI_FOLDER):
    """Coregister electrodes to MRI. This function opens a GUI in which the
    user can manually adjust the position of the electrodes. You should save
    the trans file in the GUI as fonov-trans.fif for the rest of the pipeline
    to work.

    Parameters
    ----------
    fname_electrodes : str
        Path to file containing electrode positions.
    subj : str
        Subject name.
    mri_folder : str
        Path to MRI folder.
    """
    # Make directory if necessary
    os.makedirs(mri_folder, exist_ok=True)

    # perform co-registration with gui
    mne.viz.set_3d_backend('pyvistaqt')
    mne.gui.coregistration(
        inst=fname_electrodes, subjects_dir=mri_folder, subject=subj,
        mark_inside=True, head_opacity=0.5, block=True)


def verify_coregistration(
        mri_folder=MRI_FOLDER, subj=PARCELLATION_SUBJ,
        trans_fname=LF_TRANS_FNAME):
    """Verify that coregistration is accurate.

    Parameters
    ----------
    mri_folder : str
        Path to MRI folder.
    subj : str
        Subject name.
    trans_fname : str
        Path to trans file created in coregistration step.
    """
    # Read in trans file
    trans = mne.read_trans(trans_fname)

    # Verify that the trans-file is doing its job and electrodes are aligned
    # with Freesurfer computed surfaces
    _ = mne.viz.plot_alignment(
        raw.info, trans=trans, subject=subj, dig=False,
        eeg=["original", "projected"], coord_frame="head",
        subjects_dir=mri_folder)


def setup_source(
        subj=PARCELLATION_SUBJ, mri_folder=MRI_FOLDER, spacing=SOURCE_SPACING):
    """Setup source model.

    Parameters
    ----------
    subj : str
        Subject name.
    mri_folder : str
        Path to MRI folder.
    spacing : str
        Source spacing.

    Returns
    -------
    src : mne.SourceSpaces
        Source model.
    """
    # Setup source model
    src_fname = f"{mri_folder}/eeg_src.fif"

    # Load if already exists
    if os.path.exists(src_fname):
        return mne.read_source_spaces(src_fname)

    # Setup if necessary and save
    src = mne.setup_source_space(
        subj, spacing=spacing, add_dist=False, subjects_dir=mri_folder)
    src.save(src_fname, overwrite=True)
    return src


def compute_bem(subj=PARCELLATION_SUBJ, mri_folder=MRI_FOLDER):
    """Compute boundary element model.

    Parameters
    ----------
    subj : str
        Subject name.
    mri_folder : str
        Path to MRI folder.

    Returns
    -------
    bem : mne.BemSolution
        Boundary element model.
    """
    # Compute boundary element model
    bem_fname = f"{mri_folder}/eeg_bem.fif"

    # Load if already exists
    if os.path.exists(bem_fname):
        return mne.read_bem_solution(bem_fname)

    # Calculate and save if necessary
    model = mne.make_bem_model(subject=subj, subjects_dir=mri_folder)
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_fname, bem, overwrite=True)
    return bem


def compute_leadfield(
        raw, src, bem, mri_folder=MRI_FOLDER, trans_fname=LF_TRANS_FNAME):
    """Compute leadfield from set-up source model and boundary element model
    (BEM). Source model and BEM must be computed before running this function.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data.
    src : mne.SourceSpaces
        Source model.
    bem : mne.BemSolution
        Boundary element model.
    mri_folder : str
        Path to folder containing MRI data.
    trans_fname : str
        Path to trans file created in coregistration step.
    """
    # Read in trans file
    trans = mne.read_trans(trans_fname)

    # Compute lead field
    fwd_fname = f"{mri_folder}/eeg_fwd.fif"
    fwd = mne.make_forward_solution(
        raw.info, trans, src, bem, mindist=5.0, meg=False)
    mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

    # save the forward model with the dipoles fixed to normal surces
    fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
    mne.write_forward_solution(fwd_fname, fwd_fixed, overwrite=True)


if __name__ == '__main__':
    # Download fsaverage files
    mne.set_config('SUBJECTS_DIR', MRI_FOLDER)

    # Read in electrode positions from a file
    fname_electrodes = f'{MNE_RAW_FOLDER}/NDARAA075AMK-raw.fif'
    raw = mne.io.read_raw_fif(fname_electrodes)

    # %% Perform coregistration
    coregistration(fname_electrodes)

    # %% Verify that coregistration is accurate
    verify_coregistration()

    # %% Set up source model
    src = setup_source()

    # %% Compute BEM
    bem = compute_bem()

    # %% Compute leadfield
    compute_leadfield(raw, src, bem)
