import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import mne
from pathlib import Path
import os

data_path = Path("D:/pro")

# % BioSemi triggers
    # % 120: left good
    # % 122: right good
    # % 150: left bad
    # % 155: right bad
    # % 130: no response
    # % 140: left
    # % 141: right
    # % 200: cross
    # % 201: stim
    # % 202: decision
    # % 203: feedback
    # % 204: rest
    # % 205: imagery
    # % 206: pause

event_id = {'left_good': 120, 'right_good': 122, 'left_bad': 150, 'right_bad': 155, 'no_response': 130, 'left': 140, 'right': 141, 'cross': 200, 'stim': 201, 'decision': 202, 'feedback': 203, 'rest': 204, 'imagery': 205, 'pause': 206}

def find_xdf_files(data_path: Path):
    """Find all .xdf files in the data folder.
    Args:
        data_path (Path): Path to the data folder.
    Returns:
        xdf_files (list): List of all .xdf files in the data folder."""
    xdf_files = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for file in filenames:
            if Path(file).suffix == ".xdf":
                xdf_files.append(Path(dirpath) / Path(file))
    return xdf_files

def create_info(data: list) -> mne.Info:
    """Create the info dictionary for the raw object.
    Args:
        data (list): List of dictionaries with the xdf data.
    Returns:
        info (mne.Info): Info dictionary for the raw object."""

    ch_names = []
    ch_types = []
    for i in range(int(data[1]['info']['channel_count'][0])):
        ch_names.append((data[1]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0]))
        ch_types.append(str.lower(data[1]['info']['desc'][0]['channels'][0]['channel'][i]['type'][0]))

    s_freq = int(data[1]['info']['nominal_srate'][0])

    ch_names = ch_names[1:]
    ch_types = ch_types[1:]

    for i in range(len(ch_names)):
        if ch_types[i] == 'eeg':
            ch_types[i] = 'eeg'
        elif ch_types[i] == 'eog':
            ch_types[i] = 'eog'
        else:
            ch_types[i] = 'misc'
    
    info = mne.create_info(ch_names, s_freq, ch_types)
    return info

def add_trigger_channel(raw: mne.io.Raw, data: list) -> mne.io.Raw:
    """Add the trigger channel to the raw object.
    Args:
        raw (mne.io.Raw): Raw object with the EEG data.
        data (list): List of dictionaries with the xdf data.
    Returns:
        raw (mne.io.Raw): Raw object with the trigger channel added."""
    event_samples = ((data[0]['time_stamps']-data[0]['time_stamps'][0])*512).astype(int)
    events = np.zeros((len(raw.times), 1))

    
    for i in range(len(event_samples)):
        events[event_samples[i]] = event_id[data[0]['time_series'][i][0]]
    
    event_info = mne.create_info(['STI'], raw.info['sfreq'], ['stim'])
    event_raw = mne.io.RawArray(events.T, event_info)
    return raw.add_channels([event_raw], force_update_info=True)

def add_electrode_locations(raw: mne.io.Raw, data: list) -> mne.io.Raw:
    """Add electrode locations to the raw object.
    Args:
        data (list): List of dictionaries with the xdf data.
        raw (mne.io.Raw): Raw object with the EEG data.
    Returns:
        raw (mne.io.Raw): Raw object with the electrode locations added."""
    montage = mne.channels.make_standard_montage('biosemi64')
    new_names = dict(zip(raw.ch_names, montage.ch_names))
    raw.rename_channels(new_names)
    raw.set_montage(montage, on_missing='ignore')
    return raw


def load_xdf_to_raw(xdf_file_path: Path) -> mne.io.Raw:
    """Load an xdf file to a mne raw object.
    Args:
        xdf_file_path (Path): Path to the xdf file.
    Returns:
        raw (mne.io.Raw): Raw object with the EEG data."""
    
    data, header = pyxdf.load_xdf(xdf_file_path)
    info = create_info(data)
    raw = mne.io.RawArray(data[1]["time_series"].T[1:]*1e-6, info)
    raw = add_trigger_channel(raw, data)
    raw = add_electrode_locations(raw, data)
    return raw



