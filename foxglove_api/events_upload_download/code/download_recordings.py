from pathlib import Path
import os

from api_utils import fg_client, download_recording

ROOT_FOLDER = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DEVICE = "nova_carter_galileo"
DOWNLOAD_FOLDER = Path(os.path.join(ROOT_FOLDER, "download"))
DOWNLOAD_FOLDER.mkdir(exist_ok=True)


def download_all_recordings(device: str = DEVICE) -> None:
    """
    Docstring for download_all_recordings

    :param device: Name of the device to download recordings from
    :type device: str
    """
    recordings = fg_client.get_recordings(device_name=device)
    for recording in recordings:
        print(f"Downloading recording ID: {recording['id']}")
        download_recording(
            recording["id"],
            recording["size"],
            Path(os.path.join(DOWNLOAD_FOLDER, f"{recording['path']}")),
        )


if __name__ == "__main__":
    print("=== Downloading recording from Foxglove")
    download_all_recordings(DEVICE)
    print("=== Downloading DONE")
