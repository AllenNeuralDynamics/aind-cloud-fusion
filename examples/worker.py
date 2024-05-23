"""
Runs fusion from config file generated
from dispim or exaspim scheduler.
"""

import glob
import os
import time
import uuid
from pathlib import Path

import yaml

import aind_cloud_fusion.blend as blend
import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.geometry as geometry
import aind_cloud_fusion.io as io
import aind_cloud_fusion.script_utils as script_utils


def execute_job(yml_path: str, xml_path: str, output_path: str):
    """
    yml_path: Local yml path
    xml_path: Local xml path
    output_path: Local results path
    """

    # Parse information from worker yaml
    # (See scheduler.py data contract)
    configs = script_utils.read_config_yaml(yml_path)
    input_path = configs["input_path"]
    output_s3_path = configs["output_path"]
    dataset_type = configs["dataset_type"]
    channel = int(configs["channel"])
    worker_cells = [tuple(cell) for cell in configs["worker_cells"]]

    # Initialize Application Objects
    # Application Object: DATASET
    xml_path = str(Path(xml_path))
    s3_path = input_path
    if dataset_type == "BigStitcherDataset":
        dataset = io.BigStitcherDataset(
            xml_path, s3_path, datastore=0
        )  # NOTE: Please select your desired datastore
    elif dataset_type == "BigStitcherDatasetChannel":
        dataset = io.BigStitcherDatasetChannel(
            xml_path, s3_path, channel, datastore=0
        )  # NOTE: Please select your desired datastore
    DATASET = dataset

    # Application Object: OUTPUT_PARAMS
    OUTPUT_PARAMS = io.OutputParameters(
        path=output_s3_path,
        chunksize=(
            1,
            1,
            128,
            128,
            128,
        ),  # NOTE: Please select your output chunk size
        resolution_zyx=(
            1.0,
            0.748,
            0.748,
        ),  # NOTE: Please select your output resolution
        datastore=0,  # NOTE: Please select your desired datastore
    )

    # Application Object: RUNTIME_PARAMS
    RUNTIME_PARAMS = io.RuntimeParameters(
        option=1,
        pool_size=int(os.environ.get("CO_CPUS", 1)),
        worker_cells=worker_cells,
    )

    # Application Parameter: CELL_SIZE
    CELL_SIZE = [128, 128, 128]  # NOTE: Please set this to = output chunk size

    # Application Parameter: POST_REG_TFMS
    POST_REG_TFMS: list[geometry.Affine] = (
        []
    )  # NOTE: Please add optional post-reg transforms

    # Application Object: BLENDING_MODULE
    BLENDING_MODULE = (
        blend.MaxProjection()
    )  # NOTE: Please choose your desired blending

    # Run fusion
    fusion.run_fusion(
        DATASET,
        OUTPUT_PARAMS,
        RUNTIME_PARAMS,
        CELL_SIZE,
        POST_REG_TFMS,
        BLENDING_MODULE,
    )

    # Log 'done' file for next capsule in pipeline.
    # Unique log filename
    unique_id = str(uuid.uuid4())
    timestamp = int(time.time() * 1000)
    unique_file_name = str(
        Path(output_path) / f"file_{timestamp}_{unique_id}.yml"
    )

    log_content = {}
    log_content["output_path"] = OUTPUT_PARAMS.path
    log_content["resolution_zyx"] = list(OUTPUT_PARAMS.resolution_zyx)

    with open(unique_file_name, "w") as file:
        yaml.dump(log_content, file)


if __name__ == "__main__":
    directory_to_search = "../data/"
    yml_files = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(directory_to_search)
        for file in files
        if file.endswith(".yml") and not file.startswith("s3")
    ]
    yml_path = str(yml_files[0])

    xml_paths = glob.glob("../data/**/*.xml")
    if len(xml_paths) == 0:
        directory_to_search = "../data/"
        xml_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(directory_to_search)
            for file in files
            if file.endswith((".xml"))
        ]
    xml_path = str(xml_paths[0])

    output_path = str(os.path.abspath("../results"))

    print(f"{yml_path=}")
    print(f"{xml_path=}")
    print(f"{output_path=}")

    execute_job(yml_path, xml_path, output_path)
