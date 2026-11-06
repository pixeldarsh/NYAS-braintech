import os
import urllib.request

data_dir = "sleep_edf"
os.makedirs(data_dir, exist_ok=True)

base_url = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

participants = [
    ("SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf"),
    ("SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf"),
    ("SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"),
    ("SC4031E0-PSG.edf", "SC4031EC-Hypnogram.edf"),
    ("SC4041E0-PSG.edf", "SC4041EC-Hypnogram.edf")
]

for pair in participants:
    for filename in pair:
        path = os.path.join(data_dir, filename)

        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, path)
        else:
            print(f"Already downloaded: {filename}")

print("\nDataset download complete!")
print("Total files:", len(os.listdir(data_dir)))