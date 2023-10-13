"""
Run necessary FreeSurfer functions to translate T1 brain template to surfaces
that are necessary for source estimation using MNE.

We use following template, Fonov et al, a pediatric template generated from a
representative sample of the american population (4.5 - 18.5 y.o.)
https://www.bic.mni.mcgill.ca/~vfonov/nihpd/obj1/nihpd_sym_04.5-08.5_nifti.zip
"""
# Import necessary modules
import subprocess
import os
from core.params import *


def run_recon_all(
        freesurfer_folder=FREESURFER_FOLDER, mri_folder=MRI_FOLDER,
        subj=PARCELLATION_SUBJ):
    """Run FreeSurfer's recon-all function to reconstruct surfaces.

    Parameters
    ----------
    freesurfer_folder : str
        Path to FreeSurfer folder.
    mri_folder : str
        Path to MRI folder.
    subj : str
        Subject name.
    """
    # Run reconstruction with FreeSurfer
    command = f'''export FREESURFER_HOME={freesurfer_folder};
    source $FREESURFER_HOME/SetUpFreeSurfer.sh;
    export SUBJECTS_DIR={mri_folder};

    recon-all -all -subjid {subj} -i \
        {mri_folder}/nihpd_sym_04/nihpd_sym_04.5-08.5_t1w.nii'''

    # Enable resumption of partially processed data
    if os.path.exists(f'{mri_folder}/{subj}'):
        semicolon_split = command.split(';')
        no_i = semicolon_split[3].split(' -i')[0]
        semicolon_split[3] = no_i
        command = ';'.join(semicolon_split)

    # Remove is running script that can block recon-all resumption if recon-all
    # quit unexpectedly
    is_running_f = f'{mri_folder}/{subj}/scripts/IsRunning.lh+rh'
    if os.path.exists(is_running_f):
        os.remove(is_running_f)

    # Print FreeSurfer output live as it is generated
    with subprocess.Popen(
            command, stdout=subprocess.PIPE, shell=True, bufsize=1,
            universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')


def make_watershed_bem(
        freesurfer_folder=FREESURFER_FOLDER, mri_folder=MRI_FOLDER,
        subj=PARCELLATION_SUBJ):
    """Create BEM surfaces using FreeSurfer; first with T1 image to give
    acceptable surfaces for the outer skull and outer skin, and then with
    brain.mgz to give a good inner skull surfaces.

    Parameters
    ----------
    freesurfer_folder : str
        Path to FreeSurfer folder.
    mri_folder : str
        Path to MRI folder.
    subj : str
        Subject name.
    """
    # Run watershed algorithm with FreeSurfer
    command = f'''export FREESURFER_HOME={freesurfer_folder};
    source $FREESURFER_HOME/SetUpFreeSurfer.sh;
    export SUBJECTS_DIR={mri_folder};

    mne watershed_bem -s {subj};

    mv {mri_folder}/{subj}/bem {mri_folder}/{subj}/bem_tmp;

    mne watershed_bem -s {subj} -v {mri_folder}/{subj}/mri/brain.mgz -a;

    mv -f {mri_folder}/{subj}/bem_tmp/outer_*.surf {mri_folder}/{subj}/bem;

    mv -f {mri_folder}/{subj}/bem_tmp/watershed/fonov_outer_* \
        {mri_folder}/{subj}/bem/watershed;

    rm -rf {mri_folder}/{subj}/bem_tmp'''

    # Print FreeSurfer output live as it is generated
    with subprocess.Popen(
            command, stdout=subprocess.PIPE, shell=True, bufsize=1,
            universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')


if __name__ == '__main__':
    run_recon_all()
    make_watershed_bem()
