import os
import traceback
from pathlib import Path
from requests.exceptions import HTTPError

from foxglove.client import Client
from dotenv import load_dotenv


ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(ENV_PATH)  # take environment variables from .env file
API_TOKEN = os.getenv("FOXGLOVE_API")

fg_client = Client(API_TOKEN)


def create_device(name: str, properties: dict = None) -> dict[str]:
    """
    Creates a device on Foxglove if it does not already exist.

    :param name: Name of the device to create
    :type name: str
    :param properties: Properties of the device
    :type properties: dict
    :return: Device information
    :rtype: dict[str, Any]
    """

    # Check if device already exists
    try:
        device = fg_client.get_device(device_name=name)
        if len(device):
            return device
    except HTTPError:
        print("Device does not exist, creating new one...")

    # If not available, create and return new device
    try:
        device = fg_client.create_device(name=name, properties=properties)
        print(f"Device {name} created")
        return device
    except HTTPError:
        print(traceback.format_exc())
    return {}


def upload_file_to_foxglove(filepath: str, device_name: str) -> None:
    """
    Uploads a recording to Foxglove under the specified device.

    :param filepath: Path to the recording file
    :type filepath: str
    :param device_name: Name of the device to upload the recording to
    :type device_name: str
    """
    filename = os.path.basename(filepath)
    print(f"Uploading file {filename} to device {device_name}")
    with Path(filepath).open("rb") as byte_stream:
        fg_client.upload_data(
            device_name=device_name,
            filename=filename,
            data=byte_stream,
            callback=lambda size, progress: print(
                f"{progress / size * 100:.2f}% uploaded"
            ),
        )


def download_recording(
    recording_id: str,
    recording_size: int,
    output_filepath: str,
    attachments: bool = False,
) -> None:
    """
    Downloads a recording from Foxglove.

    :param recording_id: ID of the recording to download
    :type recording_id: str
    :param recording_size: Size of the recording in bytes
    :type recording_size: int
    :param output_filepath: Path to save the downloaded recording
    :type output_filepath: str
    :param attachments: Whether to include attachments in the download
    :type attachments: bool
    """
    print(f"Downloading recording {recording_id} to {output_filepath}")
    try:
        recording_data = fg_client.download_recording_data(
            id=recording_id,
            include_attachments=attachments,
            callback=lambda progress: print(f"{progress / recording_size * 100:.2f}%"),
        )
        with Path(output_filepath).open("wb") as byte_stream:
            byte_stream.write(recording_data)
    except HTTPError:
        print(f"Could not download recording: {recording_id}")


def delete_all_events_from_device(device_name: str) -> None:
    """
    Delete all events from a device.

    :param device_name: Name of the device to delete events from
    :type device_name: str
    """
    try:
        events = fg_client.get_events(device_name=device_name)
        for event in events:
            fg_client.delete_event(event_id=event.id)
        print(f"All events from device {device_name} deleted")
    except HTTPError:
        print(traceback.format_exc())


def delete_device(device_name: str) -> None:
    """
    Delete a device.

    :param device_name: Name of the device to delete
    :type device_name: str
    """
    # First delete all recordings from the device
    delete_recordings(device_name)
    # Then delete the device itself
    try:
        fg_client.delete_device(device_name=device_name)
        print(f"Device {device_name} deleted")
    except HTTPError:
        print(traceback.format_exc())


def delete_recordings(device_name: str) -> None:
    """
    Delete all recordings from a device.

    :param device_name: Name of the device to delete recordings from
    :type device_name: str
    """
    try:
        recordings = fg_client.get_recordings(device_name=device_name)
        for recording in recordings:
            fg_client.delete_recording(recording_id=recording["id"])
        print(f"All recordings from device {device_name} deleted")
    except HTTPError:
        print(traceback.format_exc())


if __name__ == "__main__":
    delete_recordings("nova_carter_galileo")  # For testing purposes only
    delete_device("nova_carter_galileo")  # For testing purposes only
