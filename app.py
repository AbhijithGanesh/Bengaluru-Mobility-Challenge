import logging
import os
import sys
import warnings
from pathlib import Path

import cv2
from index import Solution
from PrenAbhi import global_logger as logger

cv2.setLogLevel(0)
warnings.filterwarnings("ignore")


def process_camera(solution: Solution, camera: str, model_file: str, output_file: str):
    """Processes data for a specific camera."""
    try:
        input_file = f"results_{camera}.hd5"
        solution.aggregate_all_processes(input_file, model_file, output_file)
        logging.info(f"Processed {camera} data successfully.")
    except Exception as e:
        logging.error(f"Error processing {camera} data: {str(e)}")


def validate_input_file(input_file: Path) -> bool:
    """Validates the existence of the input file."""
    if "json" not in input_file:
        logging.error("Input file must be a JSON file.")
        return False
    return True


def create_directories_submission(team_name: str, classes: list[str]):

    os.makedirs(f"data/{team_name}", exist_ok=True)
    os.makedirs(f"data/{team_name}/Matrices", exist_ok=True)
    for class_name in classes:
        os.makedirs(f"data/{team_name}/Images/{class_name}", exist_ok=True)


def main():
    args = sys.argv

    try:
        input_file = args[1]
        team_name = args[2]
    except IndexError:
        logger.fatal("Input file not provided.")
        return

    team_name = "ChennaiMobility"
    classes = ["Bus", "Bicycle", "Car", "LCV", "Truck", "Two-Wheeler", "Three-Wheeler"]
    create_directories_submission(team_name, classes)

    if not validate_input_file(input_file):
        logger.error("Invalid input file.")
        return
    soln = Solution(model="models/prenabhi-noaug-30.pt", team_name=team_name)
    cameras = soln.process_input_json(input_file)
    logging.info("Processing video...")
    soln.process(cameras)
    logging.info("Video processing completed successfully.")
    logging.info("All processing completed successfully.")


if __name__ == "__main__":
    main()
