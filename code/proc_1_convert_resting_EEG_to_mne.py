"""Convert downloaded resting-state EEG files to MNE format for analysis.
"""
# %% Import modules
import mne
import pandas as pd
from core.params import *


def convert_raw_to_mne_one_subj(
        i, subject, subjects_df, mne_raw_folder=MNE_RAW_FOLDER,
        download_folder=DOWNLOAD_FOLDER, data_csvs=DATA_CSVS,
        montage_fname=MONTAGE_FNAME, s_freq=S_FREQ,
        rest_chanlocs_csv=REST_CHANLOCS_CSV, event_ids=EVENT_IDS,
        electrode_positions=ELECTRODE_POSITIONS, highpass_freq=HIGHPASS_FREQ):
    """Convert one subject's resting-state EEG data to MNE format for analysis.

    Parameters
    ----------
    i : int
        Index of subject.
    subject : str
        Subject ID.
    subjects_df : pandas DataFrame
        DataFrame with subject information.
    mne_raw_folder : str
        Path to folder to save converted raws.
    download_folder : str
        Path to folder with downloaded data.
    data_csvs : list of str
        List of CSV files to convert.
    montage_fname : str
        Path to montage file.
    s_freq : int
        Sampling frequency, in Hz.
    rest_chanlocs_csv : str
        Path to CSV file with channel locations for resting-state data.
    event_ids : dict
        Dictionary with event IDs.
    electrode_positions : str
        Path to file with electrode positions.
    highpass_freq : int
        High-pass filter frequency, in Hz.

    Returns
    -------
    err_strs : list of str
        List of error strings encountered during download.
    """
    print("\nSubject %s (%d/%d)" % (subject, i + 1, len(subjects_df.EID)))

    # Iterate through each block
    raws, err_strs = [], []
    file_name = f"{mne_raw_folder}/{subject}-raw.fif"
    data_path = f"{download_folder}/{subject}/EEG/csv_format/"
    for data_csv, event_csv in zip(*[iter(data_csvs)] * 2):
        # Determine file name for data
        data_fname = data_path + data_csv

        # See if data is present
        if not (os.path.exists(data_fname)):
            print("raw is not found")
            return err_strs

        # do nothing if the converted file already exists
        if os.path.exists(file_name):
            print("raw is already converted!")
            return err_strs

        # -- read in data file ------------------------------------------------
        try:
            dat = np.loadtxt(data_fname, delimiter=",")
        except ValueError:
            err_str = 'Reading in data file'.upper()
            err_strs.append(err_str)
            continue

        # Make montage
        sfp_montage = mne.channels.read_custom_montage(montage_fname)
        info = mne.create_info(sfp_montage.ch_names, s_freq, "eeg")
        info.set_montage(sfp_montage)

        # Read in channel information
        chan_locs = pd.read_csv(rest_chanlocs_csv)
        if dat.shape[0] < len(info.ch_names):
            try:
                info.pick_channels(chan_locs['labels'].to_list(), ordered=True)
            except AttributeError:
                err_str = 'Picking channels with Info object'.upper()
                err_strs.append(err_str)
                continue

        # Make MNE raw object
        try:
            raw = mne.io.RawArray(dat, info)
        except ValueError:
            err_str = 'Making MNE Raw object'.upper()
            err_strs.append(err_str)
            continue
        raw.pick_channels(chan_locs['labels'].to_list(), ordered=True)

        stim_info = mne.create_info(
            ch_names=["stim"], sfreq=s_freq, ch_types="stim"
        )
        stim_raw = mne.io.RawArray(np.zeros((1, len(raw.times))), stim_info)
        raw.add_channels([stim_raw], force_update_info=True)

        # -- read in triggers and clean them up -------------------------------
        try:
            df = pd.read_csv(data_path + event_csv).drop_duplicates()
        except FileNotFoundError:
            err_str = 'No event file'.upper()
            err_strs.append(err_str)
            continue
        except pd.errors.EmptyDataError:
            err_str = 'No data in event file'.upper()
            err_strs.append(err_str)
            continue
        try:
            trigger_name_length = np.array([len(t) for t in df.type])
        except TypeError:
            err_str = 'Reading in triggers'.upper()
            err_strs.append(err_str)
            continue
        df = df[trigger_name_length < max(trigger_name_length)]
        df = df[df["type"].astype("int").isin(list(
            event_ids.values()))]
        df["duration"] = np.zeros((len(df)))
        evs = df[["sample", "duration", "type"]].astype("int").to_numpy()
        try:
            raw.add_events(evs, stim_channel="stim")
        except ValueError:
            err_str = 'Adding stim channel'.upper()
            err_strs.append(err_str)
            continue

        # -- find flat channels -----------------------------------------------
        flat_chans = np.mean(raw._data, axis=1) == 0
        bad_channels = list(np.array(raw.ch_names)[flat_chans])

        # -- find channels with variance 3 SDs away from the mean -------------
        var = np.var(raw._data, axis=1)
        high_var_mask = np.abs(var - var.mean()) > (3 * var.std())
        high_var_chans = np.array(raw.ch_names)[high_var_mask].tolist()
        bad_channels.extend(high_var_chans)
        bad_channels = list(set(bad_channels))

        # -- set flat and extreme-variance channels as bad channels -----------
        raw.info["bads"] = bad_channels
        print("Bad channels: ", raw.info["bads"])

        # -- interpolate bad channels & disable re-referencing ----------------
        try:
            raw.interpolate_bads()
        except ValueError:
            err_str = 'Interpolating bads'.upper()
            err_strs.append(err_str)
            continue
        raw.pick_types(eeg=True, stim=True)
        raw.set_eeg_reference(ref_channels=[])
        positions = mne.io.read_raw_fif(electrode_positions)
        assert positions.ch_names == raw.ch_names[:-1]

        # High-pass filter to remove block switching artifacts
        raw = raw.filter(highpass_freq, None)
        raws.append(raw)

    # Concatenate raws for each block
    if raws:
        big_raw = mne.concatenate_raws(raws)
        big_raw.save(file_name, overwrite=True)

    print("")
    num_converted = len(os.listdir(mne_raw_folder))
    print(f"Number of raw converted: {num_converted}")
    return err_strs


def convert_rest_to_mne_all_subjs(
        mne_raw_folder=MNE_RAW_FOLDER, subjects_csv=ALL_SUBJECTS_CSV,
        age_max=AGE_MAX):
    """Convert all resting-state EEG files to MNE format for analysis.

    Parameters
    ----------
    mne_raw_folder : str
        Path to folder to save converted raws.
    subjects_csv : str
        Path to CSV file with all subjects.
    age_max : int
        Maximum age of subjects to convert.
    """
    # Make directory to save converted raws
    os.makedirs(mne_raw_folder, exist_ok=True)

    # Load in subjects data
    all_df = pd.read_csv(subjects_csv).query('Age < @age_max')

    # Convert each raw to MNE and report error frequencies
    err_count = {}
    for i, subject in enumerate(sorted(all_df.EID)):
        err_strs = convert_raw_to_mne_one_subj(i, subject, all_df)
        for err_str in err_strs:
            if err_str not in err_count.keys():
                err_count[err_str] = 0
            err_count[err_str] += 1
            print("\nERROR: %s (#%d)\n" % (err_str, err_count[err_str]))
        print("Error Frequencies: {}".format(err_count))


if __name__ == '__main__':
    convert_rest_to_mne_all_subjs()
