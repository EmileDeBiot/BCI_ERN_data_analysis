from load_xdf import load_xdf_to_raw, find_xdf_files, data_path
from mne.preprocessing import ICA
from pyprep.find_noisy_channels import NoisyChannels

def main():
    xdf_files = find_xdf_files(data_path)
    for file in xdf_files:
        raw = load_xdf_to_raw(file)

        # Apply bandpass filter
        raw.filter(l_freq=0.3, h_freq=40, n_jobs=-1)

        # Make a copy to perform ICA on
        r = raw.copy() 

        # Apply highpass filter to the copy
        r.filter(l_freq=1, h_freq=None, n_jobs=-1)

        # Find bad channels
        noisy = NoisyChannels(r)
        noisy.find_all_bads(r)
        bad_chs = noisy.get_bads()

        # interpolate bad channels
        raw.info['bads'] = bad_chs
        r.info['bads'] = bad_chs
        raw.interpolate_bads()
        r.interpolate_bads()

        # Perform ICA
        ica = ICA(n_components=20, random_state=97, max_iter='auto')
        ica.fit(r)

        # Plot ICA components
        

