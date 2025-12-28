import os
from pathlib import Path
import math
from datetime import datetime
import pytz
from requests.exceptions import HTTPError

import numpy as np
from PIL import Image
import yaml

from foxglove import open_mcap, Channel
from foxglove.channels import (
    FrameTransformChannel,
    CompressedImageChannel,
    PosesInFrameChannel,
    GridChannel,
)
from foxglove.schemas import (
    FrameTransform,
    CompressedImage,
    PosesInFrame,
    Grid,
    Vector2,
    Pose,
    Quaternion,
    Timestamp,
    Vector3,
    PackedElementField,
    PackedElementFieldNumericType,
)

from api_utils import (
    fg_client,
    create_device,
    upload_file_to_foxglove,
)


ROOT_FOLDER = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_FOLDER = Path(os.path.join(ROOT_FOLDER, "data"))
IMAGES_FOLDER = Path(os.path.join(DATA_FOLDER, "rosbag_mapping_data"))
OUTPUT_FOLDER = Path(os.path.join(ROOT_FOLDER, "output"))
OUTPUT_FOLDER.mkdir(exist_ok=True)
DEVICE = "nova_carter_galileo"
MCAP_FILEPATH = Path(os.path.join(OUTPUT_FOLDER, f"{DEVICE}.mcap"))

YAW_THRESHOLD_DEGREES = 45.0


def quaternion_to_rpy_deg(x: float, y: float, z: float, w: float) -> list[float]:
    """
    Convert a quaternion to roll, pitch, yaw in degrees.

    :param x: x component of the quaternion
    :type x: float
    :param y: y component of the quaternion
    :type y: float
    :param z: z component of the quaternion
    :type z: float
    :param w: w component of the quaternion
    :type w: float
    :return: Roll, pitch, yaw in degrees
    :rtype: list[float]
    """
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(t0, t1))

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.degrees(math.asin(t2))

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(t3, t4))

    return [roll, pitch, yaw]


def publish_occupancy_grid_message(
    config_path: str,
    image_path: str,
    ts: float = 0,
) -> None:
    """
    Publish Grid message from occupancy map image and config file.

    :param config_path: Path to the occupancy grid config file
    :type config_path: str
    :param image_path: Path to the occupancy grid image file
    :type image_path: str
    :param ts: Timestamp for the message
    :type ts: float
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    origin = config["origin"]
    resolution = config["resolution"]
    image = Image.open(image_path).convert("L")
    # The occupancy grid image is stored upside down, so we need to flip it
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    image_data = np.array(image)
    _, width = image_data.shape

    # Flatten the grid data
    data = image_data.flatten().astype(np.uint8).tobytes()

    grid = Grid(
        timestamp=Timestamp(sec=0, nsec=0).from_epoch_secs(float(ts)),
        frame_id="map",
        pose=Pose(
            position=Vector3(
                x=origin[0],
                y=origin[1],
                z=0,
            ),
            orientation=Quaternion(
                x=0,
                y=0,
                z=0,
                w=1,
            ),
        ),
        cell_size=Vector2(x=resolution, y=resolution),
        column_count=width,
        cell_stride=1,
        row_stride=width * 1,
        fields=[
            PackedElementField(
                name="occupancy", offset=0, type=PackedElementFieldNumericType.Uint8
            )
        ],
        data=data,
    )

    channel = GridChannel(topic="/map")
    timestamp = int(ts.replace(".", "")[0:19])
    channel.log(grid, log_time=timestamp)


def publish_images(images_folder: str) -> None:
    """
    Generate and publish a video stream from a folder of images.

    :param images_folder: Folder containing the images. It will also serve as the topic name.
    :type images_folder: str
    """
    filepath = os.path.join(IMAGES_FOLDER, images_folder)
    channel = CompressedImageChannel(topic="/" + images_folder)
    images = os.listdir(filepath)
    images.sort()
    print(f"Generating video stream for camera {images_folder}")
    last_percentage = 0
    for i, image in enumerate(images):
        with open(os.path.join(filepath, image), mode="rb") as img_file:
            img = img_file.read()
        image = image.replace(".jpeg", "")
        nsecs = int(image)
        secs = float(image) / 1e9
        ts = Timestamp(sec=0).from_epoch_secs(secs)
        comp_img = CompressedImage(
            timestamp=ts, frame_id="base_link", data=img, format="jpeg"
        )
        channel.log(comp_img, log_time=nsecs)
        percentage = int((i + 1) / len(images) * 100)
        if percentage - last_percentage > 10:
            print(f"  Logged image {(i + 1) / len(images) * 100:.0f}%")
            last_percentage = percentage


def publish_pose_from_tum(tum_filepath: Path, yaw_threshold: float) -> None:
    """
    Publish pose and path from a TUM format file.
    This will also create events when there is a significant change in yaw.

    :param tum_filepath: Path to the TUM file
    :type tum_filepath: Path
    :param yaw_threshold: Threshold in degrees to create an event on yaw change
    :type yaw_threshold: float
    """
    with open(tum_filepath, encoding="utf-8") as f:
        lines = f.readlines()
    frame_channel = FrameTransformChannel("/tf")
    yaw_channel = Channel("/rpy")
    poses_channel = PosesInFrameChannel("/path")
    poses = []
    prev_angle = 0
    for line in lines:
        elements = line.split(" ")
        ts = elements[0]
        timestamp = Timestamp(sec=0, nsec=0).from_epoch_secs(float(ts))
        translation = Vector3(
            x=float(elements[1]), y=float(elements[2]), z=float(elements[3])
        )
        rotation = Quaternion(
            x=float(elements[4]),
            y=float(elements[5]),
            z=float(elements[6]),
            w=float(elements[7]),
        )
        tf = FrameTransform(
            parent_frame_id="map",
            child_frame_id="base_link",
            timestamp=timestamp,
            translation=translation,
            rotation=rotation,
        )
        pose = Pose(
            position=translation,
            orientation=rotation,
        )
        poses.append(pose)
        poses_in_frame = PosesInFrame(frame_id="map", poses=poses, timestamp=timestamp)

        timestamp = int(ts.replace(".", "")[0:19])
        frame_channel.log(tf, log_time=timestamp)
        poses_channel.log(poses_in_frame, log_time=timestamp)

        rpy = quaternion_to_rpy_deg(
            x=float(elements[4]),
            y=float(elements[5]),
            z=float(elements[6]),
            w=float(elements[7]),
        )
        yaw_channel.log(
            {"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}, log_time=timestamp
        )

        publish_occupancy_grid_message(
            "data/occupancy_map.yaml", "data/occupancy_map.png", ts=ts
        )

        if abs(prev_angle - rpy[2]) > yaw_threshold:
            print("Writing event")
            start = datetime.fromtimestamp(float(ts), tz=pytz.UTC)
            try:
                fg_client.create_event(
                    device_name=DEVICE,
                    start=start,
                    end=None,
                    metadata={"direction (deg)": f"{rpy[2]:.1f}"},
                )
            except HTTPError:
                print("Error creating event")
            prev_angle = rpy[2]


def main() -> None:
    writer = open_mcap(MCAP_FILEPATH, allow_overwrite=True)

    create_device(DEVICE)

    print("=== Generating trayectory")
    publish_pose_from_tum(
        DATA_FOLDER.joinpath("training_trajectory_poses.tum"), YAW_THRESHOLD_DEGREES
    )
    print("=== Trayectory DONE")

    print("=== Generating images")
    publish_images("back_stereo_camera_right")
    publish_images("back_stereo_camera_left")
    publish_images("front_stereo_camera_right")
    publish_images("front_stereo_camera_left")
    publish_images("left_stereo_camera_right")
    publish_images("left_stereo_camera_left")
    publish_images("right_stereo_camera_right")
    publish_images("right_stereo_camera_left")
    print("=== Images DONE")

    writer.close()

    print("=== Uploading recording to Foxglove")
    upload_file_to_foxglove(str(MCAP_FILEPATH), DEVICE)
    print("=== Uploading DONE")


if __name__ == "__main__":
    main()
