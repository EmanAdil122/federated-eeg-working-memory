
from pathlib import Path
import mne

DATA_DIR = Path("data")

for eeg_file in sorted(DATA_DIR.glob("sub-*/eeg/*_eeg.set")):
    print("\n" + "=" * 60)
    print(eeg_file.parent.parent.name)
    print("=" * 60)

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=False,
        verbose="ERROR"
    )

    print("Total channels:", len(raw.ch_names))
    print("\nChannel names and types:")

    channel_types = raw.get_channel_types()

    for name, channel_type in zip(raw.ch_names, channel_types):
        print(f"{name:10} -> {channel_type}")

    print("\nChannel type counts:")
    for channel_type in sorted(set(channel_types)):
        print(
            f"{channel_type}: "
            f"{channel_types.count(channel_type)}"
        )

    raw.close()
