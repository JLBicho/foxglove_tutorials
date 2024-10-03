import csv
import typing
from pathlib import Path
import os
import json
from inflection import underscore

from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding

WORKDIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(WORKDIR, "csv_headers")
SCHEMAS_DIR = os.path.join(WORKDIR, "schemas")
PATH_TO_OUTPUT_FOLDER = Path(os.path.join(WORKDIR, "mcap"))

if not os.path.exists(PATH_TO_OUTPUT_FOLDER):
    from csv_header import generate_csv_headers
    generate_csv_headers()

if not os.path.exists(PATH_TO_OUTPUT_FOLDER):
    os.mkdir(PATH_TO_OUTPUT_FOLDER)


def data_reader(csv_path: typing.Union[str, Path]):
    """ Function to read from csv file and yield each value """
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        for csv_data in csv.reader(csv_file):
            if csv_data[0] == os.path.basename(csv_path).replace(".csv", ""):
                continue
            yield csv_data[0]


def write_mcap(filename, topic_name, topic_data, schema_name="Vector3"):
    with open(os.path.join(PATH_TO_OUTPUT_FOLDER, filename+".mcap"), "wb") as f:
        writer = Writer(f)
        writer.start("x-jsonschema")
        json_filename = os.path.join(SCHEMAS_DIR, schema_name+".json")
        with open(Path(__file__).parent / json_filename, "rb") as f:
            schema = f.read()
            schema_id = writer.register_schema(
                name="foxglove."+schema_name,
                encoding=SchemaEncoding.JSONSchema,
                data=schema)
            channel_id = writer.register_channel(
                topic=topic_name,
                message_encoding=MessageEncoding.JSON,
                schema_id=schema_id)
        last_print = 0

        for i, (data) in enumerate(topic_data):
            writer.add_message(
                channel_id,
                log_time=time[i],
                data=json.dumps(data).encode("utf-8"),
                publish_time=time[i])
            current_percentage = round(i/len(topic_data)*100, 2)
            if current_percentage-last_print > 5:
                print(f"{current_percentage} %")
                last_print = current_percentage

        writer.finish()


def readVector3element(filename, topic_name, axis=None):
    print(f"Writing {filename} to {topic_name}")
    if axis is None:
        print("Specify axis!")
        return
    values = []
    for ax in axis:
        val = []
        file = filename+ax
        filepath = os.path.join(CSV_DIR, file + ".csv")
        for data in data_reader(filepath):
            val.append(data)
        values.append(val)

    vector3 = []
    for i in range(len(values[0])):
        vector3_value = {
            "x": values[0][i],
            "y": values[1][i],
            "z": values[2][i],
        }
        vector3.append(vector3_value)
    write_mcap(filename, topic_name, vector3, "Vector3")


def readVector2element(filename, topic_name, axis=None):
    print(f"Writing {filename} to {topic_name}")
    if axis is None:
        print("Specify axis!")
        return
    values = []
    for ax in axis:
        val = []
        file = filename+ax
        filepath = os.path.join(CSV_DIR, file + ".csv")
        for data in data_reader(filepath):
            val.append(data)
        values.append(val)

    vector2 = []
    for i in range(len(values[0])):
        vector2_value = {
            "x": values[0][i],
            "y": values[1][i],
        }
        vector2.append(vector2_value)
    write_mcap(filename, topic_name, vector2, "Vector2")


# Timestamp array
TIMESTAMP_FILE = "timestamp"
time = []
time_filepath = os.path.join(CSV_DIR, TIMESTAMP_FILE+".csv")
for t in data_reader(time_filepath):
    time.append(int(t.replace(',', '').split('.')[0]))

# Read 3 element variables
XYZ_FILES = [
    "estimatedCenterOfMassPosition",
    "leftFootStateEstimatorForce",
    "leftFootStateEstimatorTorque",
    "rightFootStateEstimatorForce",
    "rightFootStateEstimatorTorque",
    "stateEstimatorInWorldFramePose"
]
for base_file in XYZ_FILES:
    try:
        readVector3element(base_file, underscore(base_file),
                           axis=["X", "Y", "Z"])
    except Exception as e:
        with open("errors.log", "a", encoding="UTF-8") as logfile:
            logfile.write(f"{base_file}: {e}\n")

Q_QD_TAU_FILES = [
    "torsoYaw",
    "torsoRoll",
    "torsoPitch",
    "leftHipYaw",
    "leftHipRoll",
    "leftHipPitch",
    "rightHipYaw",
    "rightHipRoll",
    "rightHipPitch",
    "rightHipYaw",
    "rightHipRoll",
    "rightHipPitch",
    "leftAnkleRoll",
    "leftAnklePitch",
    "rightAnkleRoll",
    "rightAnklePitch",
    "leftKneePitch",
    "rightKneePitch",
    "leftShoulderYaw",
    "leftShoulderRoll",
    "leftShoulderPitch",
    "rightShoulderYaw",
    "rightShoulderRoll",
    "rightShoulderPitch",
    "leftElbowPitch",
    "rightElbowPitch",
]
for base_file in Q_QD_TAU_FILES:
    try:
        readVector3element(base_file, underscore(base_file),
                           axis=["_q", "_qd", "TauMeasured"])
    except Exception as e:
        with open("errors.log", "a", encoding="UTF-8") as logfile:
            logfile.write(f"{base_file}: {e}\n")

# Read 2 element variables
BASE_FILE = "CenterOfPressure"
readVector2element(BASE_FILE, "center_of_pressure", axis=["X", "Y"])

Q_QD_FILES = [
    "neckYaw",
    "lowerNeckPitch",
    "upperNeckPitch",
]

for base_file in Q_QD_FILES:
    try:
        readVector2element(base_file, underscore(base_file),
                           axis=["_q", "_qd"])
    except Exception as e:
        with open("errors.log", "a", encoding="UTF-8") as logfile:
            logfile.write(f"{base_file}: {e}\n")
