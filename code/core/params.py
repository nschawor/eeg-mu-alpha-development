"""All parameters for the project are defined here.
"""
# Import necessary modules
import os
import numpy as np

# Parameters for downloading and reading in data
REMOTE_PATH = 's3://anonymous@s3.amazonaws.com/fcp-indi/data/Projects/HBN/EEG'
DATA_CSVS = ['RestingState_data.csv', 'RestingState_event.csv']
EVENT_IDS = dict(resting_start=90, eyes_open=20, eyes_closed=30)
S_FREQ = 500  # sampling frequency, in Hz
HIGHPASS_FREQ = 1  # Hz
AGE_MIN = 5
AGE_MAX = 18

# Folders/directories
USER_FOLDER = os.path.expanduser('~')
# PARENT_FOLDER = '/Volumes/T7 Shield/mu_rhythm'
PARENT_FOLDER = '/Users/schaworonkown/projects/mu_rhythm'
DATA_FOLDER = f'{PARENT_FOLDER}/data'
DOWNLOAD_FOLDER = f'{DATA_FOLDER}/download'
MNE_RAW_FOLDER = f'{DATA_FOLDER}/mne_raw'
MRI_FOLDER = f'{DATA_FOLDER}/mri_fif'
FREESURFER_FOLDER = '/Applications/freesurfer/7.3.2'
RESULTS_FOLDER = f'{PARENT_FOLDER}/results'
FIG_FOLDER = f'{PARENT_FOLDER}/figures'
SSD_FOLDER = f'{RESULTS_FOLDER}/ssd'
SENSOR_SPARAM_FOLDER = f'{RESULTS_FOLDER}/spec_param_sensor'
SSD_SPARAM_FOLDER = f'{RESULTS_FOLDER}/spec_param_ssd'
PSD_FOLDER = f'{FIG_FOLDER}/psd'
SUBJ_QUALITY_PLTS_FOLDER = f'{FIG_FOLDER}/subj_quality_plts'
COMPARISONS_FOLDER = f'{SUBJ_QUALITY_PLTS_FOLDER}/comparisons'
COMP_TIMESERIES_FOLDER = f'{RESULTS_FOLDER}/comp_timeseries'
BYCYCLE_FOLDER = f'{RESULTS_FOLDER}/bycycle'
BURST_STATS_FOLDER = f'{RESULTS_FOLDER}/burst_stats'

# Data files
ALL_SUBJECTS_CSV = f'{DATA_FOLDER}/all_subjects.csv'
REST_CHANLOCS_CSV = f'{DATA_FOLDER}/RestingState_chanlocs.csv'
ELECTRODE_POSITIONS = f'{DATA_FOLDER}/electrode_positions-raw.fif'
DIAGNOSES_CSV = f'{USER_FOLDER}/diagnosis.csv'
MONTAGE_FNAME = f'{DATA_FOLDER}/GSN_HydroCel_129.sfp'

# CSV output files
PEAK_FREQ_CSV = f'{RESULTS_FOLDER}/subjects_with_peak_frequency.csv'
SOURCES_CSV = f'{RESULTS_FOLDER}/sources.csv'
WAVEFORM_FEATS_CSV = f'{RESULTS_FOLDER}/waveform_feats.csv'
BURSTS_VS_NO_BURSTS_PSD_CSV = f'{RESULTS_FOLDER}/bursts_vs_no_bursts_psd.csv'

# Channels for Laplacian - 2 alpha channels, then 2 mu channels
# (important for colors in plotting)
LAPLACIAN_CHANNELS = {
    'E70': ['E65', 'E66', 'E69', 'E71', 'E74', 'E75'],
    'E83': ['E75', 'E76', 'E82', 'E84', 'E89', 'E90'],
    'E36': ['E29', 'E30', 'E35', 'E37', 'E41', 'E42'],
    'E104': ['E87', 'E93', 'E103', 'E105', 'E110', 'E111']}

# SSD parameters
SSD_BIN_WIDTH = 2  # chosen bin width after checking
SSD_BIN_WIDTHS = np.arange(0.5, 5.5, 0.5)  # ranges to check
NUM_COMPONENTS = 10

# FOOOF parameters
NR_SECONDS_SPEC = 4
PEAK_FMIN, PEAK_FMAX = 6, 13
FIT_FMIN, FIT_FMAX = 2, 40
N_PEAKS = 5
PEAK_WIDTH_LIMITS = (1, 12)

# SSD parameters
SSD_BIN_WIDTH = 2  # chosen bin width after checking
NUM_COMPONENTS = 10

# Parcellation parameters
PARCELLATION_SUBJ = 'fonov'
PARCELLATION_ANNOTATION = 'fonov_HCPMMP1'
LF_TRANS_FNAME = f'{MRI_FOLDER}/fonov-trans.fif'
INNER_SKULL_THRESHOLD = -85
SOURCE_SPACING = 'oct7'
SNR_THRESHOLD = 5  # dB
PATTERN_DIST_THRESHOLD = 0.15
SNR_THRESHOLDS = np.linspace(0, 16, 9)
PATTERN_DIST_THRESHOLDS = np.linspace(0.25, 0.1, 7)
CBAR_MIN, CBAR_MID, CBAR_MAX = 0.0, 0.01, 0.1

# Bycycle parameters
PEAK_RANGE = 6
F_BANDPASS = (1, 45)
N_PROCESSES = None
BURST_KWARGS = {
    'amp_fraction_threshold': .5,
    'amp_consistency_threshold': .5,
    'period_consistency_threshold': .5,
    'monotonicity_threshold': .5,
    'min_n_cycles': 3}  # TUNED burst detection parameters
BYCYCLE_VERBOSE = True
BYCYCLE_FEATS_TO_EXTRACT = ['period', 'volt_amp', 'time_rdsym', 'time_ptsym']

# Plotting parameters
PVAL_FORMAT = {1: 'ns', 0.05: '*', 0.01: '**', 0.001: '***', 0.0001: '****'}
WAVEFORM_FEATS_TO_PLOT = {
    'frequency': 'frequency [Hz]',
    'time_rdasym': 'rise-decay asymmetry',
    'time_ptasym': 'peak-trough asymmetry',
    'amp_norm': 'normalized amplitude'}
DIAGNOSES_TO_PLOT = {
    'No Diagnosis Given': 'None',
    'ADHD-Combined Type': 'ADHD',
    'ADHD-Inattentive Type': 'ADHD',
    'ADHD-Hyperactive/Impulsive Type': 'ADHD',
    'Autism Spectrum Disorder': 'ASD'}
DIAGNOSES_PALETTE = {
    'Alpha': ['#5a4073', '#946da1', '#e4c5db'],
    'Mu': ['#1a491e', '#326c43', '#bed2a7']}
EXAMPLE_SUBJ = 'NDARNK489GNR'
EXAMPLE_CHANNEL = 'E70'
EXAMPLE_COMP = 1
EXAMPLE_REGIONS = ('L_1_ROI-lh', 'L_LO1_ROI-lh') # top to bottom on lead fields

# Subject quality plot parameters
SUBJ_QUALITY_PLTS_OVERWRITE = False
N_ROWS, N_COLS = 1, 5
ALPHA_COLOR, MU_COLOR = '#513766', '#2B623C'
TS_LEN = 1.0
SOURCE_HEMIS = ['lh', 'rh']
SOURCE_VIEWS = ['lateral', 'medial']
