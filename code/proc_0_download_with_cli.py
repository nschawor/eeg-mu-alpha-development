"""Download data from HBN S3 bucket using CLI.
"""
#%%
import subprocess
import pandas as pd
import multiprocessing as mp
from core.params import *
import numpy as np

def download_data_one_subj(
        i_sub, subject_id, n_subjects, csv_fnames=DATA_CSVS,
        remote_path=REMOTE_PATH, folder=DOWNLOAD_FOLDER):
    """Download data for one subject.

    Parameters
    ----------
    i_sub : int
        Index of subject.
    subject_id : str
        Subject ID.
    n_subjects : int
        Number of subjects.
    csv_fnames : list of str
        List of CSV files to download.
    remote_path : str
        Remote path to data.
    folder : str
        Folder where downloaded data is saved.
    """
    # Print subject before processing
    print_str = "%s (%d/%d)" % (subject_id, i_sub + 1, n_subjects)

    # Try to download each piece of data for the subject
    for csv_fname in csv_fnames:
        remote_file_path = "%s/%s/EEG/raw/csv_format/%s" % (
            remote_path, subject_id, csv_fname)
        local_file_path = "%s/%s/EEG/csv_format/" % (folder, subject_id)

        if os.path.exists(local_file_path + csv_fname):
            print_str = "\n".join((print_str, "        %s done" % csv_fname))
            continue

        os.makedirs(local_file_path, exist_ok=True)
        _ = subprocess.run(
            ["duck", "-download", remote_file_path, local_file_path],
            capture_output=True)
        if os.path.exists(local_file_path + csv_fname):
            print_str = "\n".join(
                (print_str, "\033[92m        downloading %s "
                            "successful\033[0m" % csv_fname))
        else:
            print_str = "\n".join(
                (print_str, "\033[91m        downloading %s not "
                            "successful\033[0m" % csv_fname))
    # Print progress
    print(print_str)


def download_data_all_subjects(subjects, n_processes=N_PROCESSES):
    """Download data for all subjects.

    Parameters
    ----------
    subjects : list of str
        List of subject IDs.
    n_processes : int
        Number of processes to use.
    """
    # Use multiprocessing pool to parallelize download
    args = [(i_sub, subject_id, len(subjects)) for i_sub, subject_id in
            enumerate(subjects)]
    with mp.Pool(n_processes) as pool:
        pool.starmap(download_data_one_subj, args)


def print_downloaded_stats(
        subjects_df, folder=DOWNLOAD_FOLDER, data_csvs=DATA_CSVS):
    """Print stats about downloaded data for each release.

    Parameters
    ----------
    subjects_df : pd.DataFrame
        DataFrame with subjects.
    folder : str
        Folder where data is downloaded.
    data_csvs : list of str
        List of CSV files to check.
    """
    # Get all releases
    releases = np.unique(subjects_df.release)
    event_csv = data_csvs[1]

    # Iterate over releases
    for release in releases:
        df1 = subjects_df[subjects_df.release == release]
        subj_ids = df1.EID
        cc = 0

        # Count how many subjects have event CSV file
        for subj_id in subj_ids:
            local_file_path = "%s/%s/EEG/csv_format/" % (folder, subj_id)
            if os.path.exists(local_file_path + event_csv):
                cc += 1

        # Print stats for release
        perc = cc / len(df1) * 100
        print(f"{release} {cc:03} {len(df1):03} {perc:.3f} % done")



def download_all_subjects(subjects_csv=ALL_SUBJECTS_CSV):
    """Download data for all subjects.

    Parameters
    ----------
    subjects_csv : str
        Path to CSV file with all subjects.
    """
    # Get all subjects
    df = pd.read_csv(subjects_csv)
    print(np.unique(df.release.to_list()))
    subjects_df = df.reset_index(drop=True)
    subjects = sorted(df.EID)

    # Download data
    download_data_all_subjects(subjects)

    # Print downloaded stats
    print_downloaded_stats(subjects_df)


if __name__ == '__main__':
    # Get files
    files = [f for f in os.listdir('../data/') if 'HBN' in f]
    files = np.sort(files)
    dfs = []

    # Collect all subjects into one DataFrame
    for file in files:
        print(file)
        release = file.split('_')[1]
        df = pd.read_csv(f'../data/{file}')
        df['release'] = release
        dfs.append(df)
    df_all = pd.concat(dfs)
    df_all = df_all.drop_duplicates('EID')
    print(len(df_all))
    df_all.to_csv(f'{DATA_FOLDER}/all_subjects.csv', index=None)

    # Download data for all subjects
    download_all_subjects()
