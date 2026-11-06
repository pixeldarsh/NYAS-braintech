import os
import mne

data_dir = "sleep_edf"

for filename in sorted(os.listdir(data_dir)):
    if filename.endswith("-PSG.edf"):
        path = os.path.join(data_dir, filename)

        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)

        print(f"\nParticipant: {filename}")
        print("Duration:", round(raw.times[-1] / 3600, 2), "hours")
        print("Sampling frequency:", raw.info["sfreq"], "Hz")
        print("Channels:", raw.ch_names)