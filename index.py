from comet_ml import Experiment
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, TypedDict

import chromadb
import chromadb.errors
import cv2
import h5py
import numpy as np
import torch
from chromadb.api.models.Collection import Collection
from PIL import Image
from torchvision import models, transforms
from torchreid import utils, models as torchreid_models
from ultralytics import YOLO

from PrenAbhi import Counter
from PrenAbhi import global_logger as logger


class VehicleMatch(TypedDict):
    """
    A typing class for vehicle matching results.

    Attributes:
        vehicle_id (str): The unique identifier for the vehicle.
        similar_vehicle_id (str): The identifier for a similar vehicle found.
        frame_id (int): The frame number where the match was found.
        boundingbox (List[int]): The bounding box coordinates [x1, y1, x2, y2] for the detected vehicle.
    """

    vehicle_id: str
    similar_vehicle_id: str
    frame_id: int
    boundingbox: List[int]


class Solution:
    """
    A class for processing and analyzing video data using object detection and feature extraction.

    This class provides methods for:
    - Processing videos to detect and track objects.
    - Extracting features from detected objects using a deep learning model.
    - Saving and loading results to and from HDF5 files.
    - Saving vector embeddings to a Chroma database.
    - Comparing similarities between objects across different videos.
    - Saving frames from videos with annotated bounding boxes.
    """

    def __init__(
        self,
        model: str,
        max_frames: int = -1,
        threshold: float = 0.7,
        team_name: str = "ChennaiMobility",
        transform: transforms.Compose = None,
    ) -> None:
        """
        Initialize the Solution class with common parameters and empty vector collections.

        Args:
            model (str): Path to the YOLO model for video processing.
            max_frames (int): Maximum number of frames to process. If -1, process the entire video.
            threshold (float): Similarity threshold for logging matches.
            transform (transforms.Compose): Image transformation pipeline. If None, use the default.
        """
        self.model = YOLO(model)
        self.max_frames = max_frames
        self.threshold = threshold
        self.transform = transform or self._create_image_transform()
        self.video_frames = []
        self.team_name = team_name
        self.camera_names = []
        self.data_array = []
        self.data = {}

        self.experiment = Experiment(
            api_key=os.environ.get("API_KEY"),
            project_name="submission",
            workspace="abhijithganesh",
            display_summary_level=0,
            disabled=False,
        )

        self.test_chroma_client()
        self.init_chroma_client()

    def _calculate_line_points(
        self, w: int, h: int, ratio: float
    ) -> list[tuple[int, int]]:
        """
        Calculate the line points for object tracking.

        Args:
            w (int): Width of the video frame.
            h (int): Height of the video frame.
            ratio (float): Ratio for calculating line points.

        Returns:
            list[tuple[int, int]]: List of calculated line points.
        """
        return [
            (int(w * ratio), int(h * ratio)),  # Top-left
            (int(w * (1 - ratio)), int(h * ratio)),  # Top-right
            (int(w * (1 - ratio)), int(h * (1 - ratio))),  # Bottom-right
            (int(w * ratio), int(h * (1 - ratio))),  # Bottom-left
        ]

    def _create_image_transform(self) -> transforms.Compose:
        """
        Create the image transformation pipeline.

        Returns:
            transforms.Compose: The composed image transformations.
        """
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_feature_extractor_model(self) -> torch.nn.Module:
        """
        Load and prepare the ResNet50-IBN-A model.

        Returns:
            torch.nn.Module: The loaded ResNet50-IBN-A model.
        """
        model = torch.hub.load(
            "AbhijithGanesh/IBN-Net", "resnet50_ibn_a", pretrained=True
        )
        model.eval()
        return model

    def _prepare_features(self, collection: dict) -> np.array:
        """
        Prepare feature vectors from a collection.

        Args:
            collection (dict): The collection of features.

        Returns:
            np.array: The prepared feature vectors.
        """
        return np.array(
            [value["features"] for value in collection.values() if "features" in value]
        )

    def _prepare_image(self, image_data: np.ndarray | str) -> Image.Image:
        """
        Prepare the image data for feature extraction.

        Args:
            image_data (np.ndarray | str): The image data to prepare.

        Returns:
            Image.Image: The prepared image.
        """
        if isinstance(image_data, np.ndarray):
            return Image.fromarray(image_data)
        elif isinstance(image_data, str):
            return Image.open(image_data).convert("RGB")

    def _set_dataset_attributes(
        self, dataset: h5py.Dataset, key: str, value: dict
    ) -> None:
        """
        Set attributes for HDF5 datasets.

        Args:
            dataset (h5py.Dataset): The dataset to set attributes for.
            key (str): The key identifying the dataset.
            value (dict): The value dictionary containing attributes.
        """
        dataset.attrs["vehicle_id"] = key
        dataset.attrs["vehicle"] = value.get("vehicle", "Unknown")
        dataset.attrs["frame_id"] = value.get("frame_id", -1)

    def check_video_stream(self, video1: dict, camera_name: str) -> None:
        """
        Check if the video stream has similar vehicle objects based on their feature vectors.
        Args:
        video1 (dict): The feature vector collection for the first video.
        camera_name (str): The name of the camera corresponding to the video.
        """
        results = []
        camera_base_name = Path(
            camera_name
        ).stem  

        valid_vehicles = []
        for vehicle_id, vehicle_data in video1.items():
            features = vehicle_data.get("features")
            if features is None:
                logger.warning(
                    f"No features found for vehicle {vehicle_id} in {camera_base_name}"
                )
                continue
            current_vehicle = vehicle_data.get("vehicle", "Unknown").split("#")[0]
            valid_vehicles.append((vehicle_id, current_vehicle, features[0].tolist()))

        try:
            query_embeddings = [v[2] for v in valid_vehicles]
            query_results = self.coll.query(query_embeddings=query_embeddings)
        except Exception as e:
            logger.error(f"Error querying Chroma: {e}")
            return

        for i, (vehicle_id, current_vehicle, _) in enumerate(valid_vehicles):
            for idx, (similar_id, metadata) in enumerate(
                zip(query_results["ids"][i], query_results["metadatas"][i])
            ):
                query_vehicle = metadata["vehicle"].split("#")[0]
                if similar_id != vehicle_id and query_vehicle == current_vehicle:
                    results.append(
                        {
                            "vehicle_id": vehicle_id,
                            "similar_vehicle_id": similar_id,
                            "vehicle": query_vehicle,
                            "frame_id": metadata["frame_id"],
                            "bounding_box": metadata["bounding_box"],
                            "camera_name": metadata["camera"],
                        }
                    )

        logger.info(
            f"Processing video stream: {camera_base_name} with {len(results)} matching results."
        )
        self.experiment.log_metric(f"{camera_base_name}_matching_results", len(results))
        self.process_results(results, camera_base_name)

    def extract_features(self, vector_collection: dict) -> dict:
        """
        Extract features from the processed images using the HACNN model.

        Args:
            vector_collection (dict): A dictionary containing vector collections (features and metadata) extracted from the video.

        Returns:
            dict: The updated vector collection with extracted features.
        """
        curr_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_extractor = utils.FeatureExtractor(
            model_name="osnet_x0_75",
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=False,
        )

        transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),  # Resize image to appropriate dimensions
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize
            ]
        )

        for i, (ky, val) in enumerate(vector_collection.items(), 1):
            image_data = val.get("image")
            bounding_box = val.get("bounding_box")
            if (image_data is None) or (bounding_box is None):
                logger.error(f"Error: Image or bounding box not found for {ky}")
                continue

            # Convert bounding box to integer and crop the image
            bounding_box = [int(i) for i in bounding_box]
            image_data = self._prepare_image(image_data).crop(bounding_box)

            input_tensor = self.transform(image_data).unsqueeze(0).to(device)
            image_data = transform(image_data)

            # Add batch dimension to the image
            input_tensor = image_data.unsqueeze(0).to(
                device
            )  # Add batch dimension [1, channels, height, width]

            extracted_features = feature_extractor(input_tensor)
            vector_collection[ky]["features"] = extracted_features

            if i % 100 == 0:
                logger.info(f"Extracted features for {i} items")

        logger.info(
            "Feature extraction completed in %.3f seconds.", time.time() - curr_time
        )

        self.experiment.log_metric("feature_extraction_time", time.time() - curr_time)

        return vector_collection

    def get_frame_from_video(self, frame_id: int, camera_name: str):
        """
        Retrieve a specific frame from the video by frame_id.

        Args:
            frame_id (int): The ID of the frame to retrieve.
            camera_name (str): The name of the camera corresponding to the video.

        Returns:
            numpy.ndarray: The retrieved frame as an image.
        """

        try:
            if not Path(camera_name).exists():
                raise FileNotFoundError(f"Video file not found: {camera_name}")
        except FileNotFoundError as e:
            logger.error(f"Error: Video file not found: {camera_name}")

        cap = cv2.VideoCapture(camera_name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            logger.error(
                f"Error: Could not retrieve frame {frame_id} from {camera_name}"
            )
            return None

        return frame

    def init_chroma_client(self) -> None:
        """
        Initialize the Chroma client for vector storage.

        This method establishes a connection to the Chroma database client and
        initializes a collection for storing vehicle embeddings.
        """
        try:
            self.client = chromadb.HttpClient(
                host="localhost",
                port=8000,
            )
            self.coll = self.client.get_or_create_collection(
                name="vehicle_embeddings",
            )
            logger.info("Chroma client initialized successfully.")
        except chromadb.errors.ChromaError as e:
            logger.critical(f"Chroma client initialization failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")
            raise

    def load_results(self, input_file: str) -> dict:
        """
        Load previously saved results (features and images) from an HDF5 file.

        Args:
            input_file (str): Path to the input HDF5 file.

        Returns:
            dict: The loaded vector collection.
        """
        try:
            vector_collection = {}
            with h5py.File(input_file, "r") as processed_hdf5:
                for i, key in enumerate(processed_hdf5.keys(), 1):
                    features_dataset = processed_hdf5[f"{key}/features"]
                    image_dataset = processed_hdf5[f"{key}/image"]

                    vector_collection[key] = {
                        "features": features_dataset[()],
                        "image": image_dataset[()],
                        "vehicle": features_dataset.attrs.get("vehicle", "Unknown"),
                        "frame_id": features_dataset.attrs.get("frame_id", -1),
                    }

                    if i % 100 == 0:
                        logger.info(f"Loaded {i} items from HDF5")

            logging.info(f"Results successfully loaded from {input_file}")
            return vector_collection

        except Exception as e:
            logging.error(f"Error loading results from HDF5: {str(e)}")
            return {}

    def process(self, videos: list[str]) -> None:
        """
        Aggregate all the processing steps: video processing, feature extraction, and similarity calculation.

        Args:
            videos (list[str]): List of video file paths to process.
        """
        overall_start_time = time.time()
        logger.info("Starting aggregated process")

        for i in videos:
            self.camera_names.append(i)

        for i, video_path in enumerate(videos, 1):
            video_start_time = time.time()
            self.camera_name = video_path.name.removesuffix(video_path.suffix)
            # Process the video
            vector_collection = self.process_video(video_path)

            video_processing_time = time.time() - video_start_time
            logger.telemetry(
                f"Processed video: {self.camera_name} in {video_processing_time:.2f} seconds"
            )
            self.experiment.log_metric(
                f"{self.camera_name}_video_processing_time", video_processing_time
            )

            # Extract features
            extract_features_start_time = time.time()
            vector_collection = self.extract_features(vector_collection)

            extract_features_time = time.time() - extract_features_start_time
            logger.telemetry(
                f"Extracted features for {self.camera_name} in {extract_features_time:.2f} seconds"
            )
            self.data_array.append(vector_collection)

            self.experiment.log_metric(
                f"{self.camera_name}_extract_features_time", extract_features_time
            )

            # Save to Chroma
            save_chroma_start_time = time.time()
            self.save_to_chroma(vector_collection, self.camera_names[i - 1])
            save_chroma_time = time.time() - save_chroma_start_time
            logger.telemetry(
                f"Saved vector collection to Chroma for {self.camera_name} in {save_chroma_time:.2f} seconds"
            )
            self.experiment.log_metric(
                f"{self.camera_name}_save_chroma_time", save_chroma_time
            )

            logger.info(f"Completed processing {i}/{len(videos)} videos")

        self.data = {
            "Car": [],
            "Bus": [],
            "LCV": [],
            "Truck": [],
            "Three-Wheeler": [],
            "Two-Wheeler": [],
            "Bicycle": [],
        }

        logger.info("Checking video streams for similarities")
        for ky, val in self.data.items():
            for i in range(len(videos)):
                val.append([0] * len(videos))

        for i in range(len(self.camera_names)):
            logger.info("Working on camera: ")
            self.check_video_stream(self.data_array[i], self.camera_names[i])

        logger.info("Working on matrices")
        for ky, val in self.data.items():
            for i in range(len(val)):
                self.data[ky][i][i] = 0

        for ky, val in self.data.items():
            os.makedirs(f"data/{self.team_name}/Matrices", exist_ok=True)
            with open(f"data/{self.team_name}/Matrices/{ky}.json", "w") as f:
                json.dump(val, f)

        self.experiment.log_asset_folder(f"data/{self.team_name}/Matrices")
        overall_process_time = time.time() - overall_start_time
        logger.telemetry(f"Overall processing time: {overall_process_time:.2f} seconds")

        self.experiment.log_metric("total_processing_time", overall_process_time)
        self.experiment.log_asset("application.log")

    def process_input_json(self, json_name: str) -> list[str]:
        """
        Process the input JSON file and extract paths of videos.

        Args:
            json_name (str): The path to the input JSON file.

        Returns:
            list[str]: List of video paths found in the JSON file.
        """
        with open(json_name, "r") as f:
            data = json.load(f)
        return [Path(i) for i in data.values()]

    def process_video(self, video: str) -> dict:
        """
        Process a video using the YOLO model to detect and track objects.

        Args:
            video (str): Path to the video file.

        Returns:
            dict: A dictionary containing vector collections (features and metadata) extracted from the video.
        """
        current_time = time.time()
        cap = cv2.VideoCapture(video)
        try:
            assert cap.isOpened(), "Error reading video file"
        except AssertionError as e:
            logger.error(e)
            raise

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in video: {total_frames}")

        self.experiment.log_metric("total_frames", total_frames)

        self.max_frames = total_frames

        w, h, _fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )
        ratio = 5 / 100
        line_points = self._calculate_line_points(w, h, ratio)
        counter = Counter(
            view_img=False,
            reg_pts=line_points,
            names=self.model.names,
            draw_tracks=False,
            view_out_counts=False,
            view_in_counts=False,
            line_thickness=2,
        )

        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                logger.error(
                    "Video frame is empty or video processing has been successfully completed."
                )
                break

            tracks = self.model.track(
                im0, persist=True, show=False, verbose=False, tracker="bytetrack.yaml"
            )
            im0 = counter.start_counting(im0, tracks, frame_count)
            frame_count += 1

            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")

        logger.info(
            "Video processing completed in %.2f seconds.", time.time() - current_time
        )
        cap.release()

        self.experiment.log_metric("video_processing_time", time.time() - current_time)

        return counter.get_vector_collection()

    def save_results(self, output_file: str, dictionary: dict) -> None:
        """
        Save the processed results (features and images) to an HDF5 file.

        Args:
            output_file (str): Path to the output HDF5 file.
            dictionary (dict): The dictionary containing the results to save.
        """
        try:
            with h5py.File(output_file, "w") as processed_hdf5:
                for i, (key, value) in enumerate(dictionary.items(), 1):
                    feature_dataset = processed_hdf5.create_dataset(
                        f"{key}/features", data=value["features"]
                    )
                    image_dataset = processed_hdf5.create_dataset(
                        f"{key}/image", data=value["image"]
                    )

                    self._set_dataset_attributes(feature_dataset, key, value)
                    self._set_dataset_attributes(image_dataset, key, value)

                    if i % 100 == 0:
                        logger.info(f"Saved {i} items to HDF5")

            logger.info(f"Results successfully saved to {output_file}")

            self.experiment.log_asset(output_file)

        except Exception as e:
            logging.error(f"Error saving results to HDF5: {str(e)}")

    def save_to_chroma(
        self,
        vector_collection: dict,
        camera_name: str,
        collection_name: str = "vehicle_embeddings",
    ) -> None:
        """
        Save the vector collection to a Chroma database.

        Args:
            vector_collection (dict): The vector collection to save.
            collection_name (str): The name of the collection in Chroma. Defaults to "vehicle_embeddings".
        """
        try:
            client = chromadb.HttpClient(
                host="localhost",
                port=8000,
            )
            collection: Collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,
            )

            ids = []
            embeddings = []
            metadatas = []

            for i, (vehicle_id, data) in enumerate(vector_collection.items(), 1):
                ids.append(vehicle_id)
                features = data["features"][0]
                embeddings.append(features.tolist())
                metadatas.append(
                    {
                        "camera": str(camera_name.absolute()),
                        "vehicle": data.get("vehicle", "Unknown"),
                        "frame_id": data.get("frame_id", -1),
                        "bounding_box": str(data.get("bounding_box", [])),
                    }
                )

                if i % 100 == 0:
                    logger.info(
                        f"Added {i} items to Chroma collection '{collection_name}'"
                    )

            collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

            logger.info(
                f"Vector collection successfully saved to Chroma collection '{collection_name}'."
            )

            self.experiment.log_metric("chroma_save_count", len(ids))

        except Exception as e:
            logging.error(f"Error saving vector collection to Chroma: {str(e)}")
            raise

    def test_chroma_client(self) -> None:
        """
        Test the connection to the Chroma client.

        This method checks whether the Chroma database client is accessible and
        operational by sending a heartbeat request. If it fails, it starts Chroma.
        """
        try:
            # Try to create a Chroma client and send a heartbeat request
            client = chromadb.HttpClient(
                host="localhost",
                port=8000,
            )
            client.heartbeat()
            logger.info("Chroma client connection successful.")

        except Exception as e:
            logger.critical(f"Chroma client connection failed: {str(e)}")
            logger.critical("Starting Chroma...")

            process = subprocess.Popen(
                ["chroma", "run", "--host", "0.0.0.0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            logger.info(
                "Chroma is starting in the background. Waiting for it to become available..."
            )

            for i in range(12):
                time.sleep(5)
                try:
                    client = chromadb.HttpClient(
                        host="localhost",
                        port=8000,
                    )
                    client.heartbeat()
                    logger.info("Chroma client started successfully.")
                    return
                except Exception:
                    logger.info(
                        f"Waiting for Chroma to start... ({(i + 1) * 5} seconds passed)"
                    )

            # If Chroma failed to start after waiting
            logger.error("Chroma failed to start after 30 seconds.")
            stdout, stderr = process.communicate()
            logger.error(f"Chroma start stdout: {stdout.decode()}")
            logger.error(f"Chroma start stderr: {stderr.decode()}")

            raise Exception("Failed to start Chroma.")

    def process_results(self, results: List[VehicleMatch], source_camera: str) -> None:
        """
        Process the vehicle matching results and log the matches.

        Args:
            results (List[VehicleMatch]): List of vehicle matching results.
            source_camera (str): The name of the camera where the source video was recorded.
        """
        logger.info("Processing results of " + source_camera)
        camera_names = [Path(cam).stem for cam in self.camera_names]
        src_idx = camera_names.index(Path(source_camera).stem)
        colors = [(255, 0, 0), (0, 0, 255)]

        for result in results:
            frame = self.get_frame_from_video(result["frame_id"], result["camera_name"])
            bounding_box_str = result.get("bounding_box", "")

            if bounding_box_str:
                # Parse bounding box
                bounding_box = [
                    int(float(k)) for k in bounding_box_str.strip("[]").split(",")
                ]
                x1, y1, x2, y2 = bounding_box

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=8)

                # Prepare text
                vehicle_name = Path(result.get("vehicle", "unknown"))
                text = f"{vehicle_name.name}_{result.get('similar_vehicle_id')}"

                # Set text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.75
                font_thickness = 1

                # Calculate text size and position
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_width, text_height = text_size
                text_position = (x1 + 10, y1 + 30)
                background_position = (
                    text_position[0],
                    text_position[1] - text_height,
                    text_position[0] + text_width,
                    text_position[1],
                )

                # Draw text background
                cv2.rectangle(
                    frame,
                    background_position[:2],
                    background_position[2:],
                    (255, 255, 255),
                    cv2.FILLED,
                )

                # Draw text
                color = colors[results.index(result) % len(colors)]
                cv2.putText(
                    frame,
                    text,
                    text_position,
                    font,
                    font_scale,
                    color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

                # Save annotated frame
                camera_name = Path(result.get("camera_name", ""))
                output_filename = (
                    "data"
                    / Path(self.team_name)
                    / "Images"
                    / vehicle_name.name
                    / f"{camera_name.stem}_{vehicle_name.name}_{result.get('vehicle_id')}.jpg"
                )
                output_filename.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_filename), frame)

            # Update data matrix
            camera_name = Path(result.get("camera_name", "")).stem
            self.data[result["vehicle"]][src_idx][camera_names.index(camera_name)] += 1
