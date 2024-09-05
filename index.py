import json
import logging
import os
import subprocess
import sys
import time
from typing import List, TypedDict

import chromadb
import chromadb.errors
import cv2
import h5py
import numpy as np
import torch
from chromadb.api.models.Collection import Collection
from comet_ml import Experiment
from PrenAbhi import Counter
from PrenAbhi import global_logger as logger
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


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
        threshold: float = 0.89,
        team_name: str = "APP",
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
            api_key="zQYci303khoykJAwYAgQkG2Dj",
            project_name="submission-testing",
            workspace="abhijithganesh",
            display_summary_level=0,
            disabled=True,
        )
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

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

    def _calculate_similarities(
        self, video: np.array, reference_flattened: np.array
    ) -> None:
        """
        Calculate and log similarities between video objects.

        Args:
            video (np.array): Feature vectors of the video.
            reference_flattened (np.array): Feature vectors of the reference.
        """
        ...

    def check_video_stream(self, video1: dict, camera_name: str) -> None:
        """
        Check if two video streams are similar based on their feature vectors.

        Args:
            video1 (dict): The feature vector collection for the first video.
            camera_name (str): The name of the camera corresponding to the video.
        """
        results = []
        for key, val in video1.items():
            data = val.get("features")
            # Find cosine similarity from Chroma
            res = self.coll.query(query_embeddings=[data.tolist()])
            for i in range(len(res["distances"])):
                if (
                    res["distances"][0][i] < (1 - self.threshold)
                    and res["ids"][i] != key
                ):
                    results.append(
                        {
                            "vehicle_id": key,
                            "similar_vehicle_id": res["ids"][i][0],
                            "vehicle": res["metadatas"][0][i]["vehicle"].split("#")[0],
                            "frame_id": res["metadatas"][0][i]["frame_id"],
                            "bounding_box": res["metadatas"][0][i]["bounding_box"],
                            "camera_name": res["metadatas"][0][i]["camera"],
                        }
                    )
        logger.critical("Processing video stream : " + camera_name)
        self.process_results(results, camera_name)

    def extract_features(self, vector_collection: dict) -> dict:
        """
        Extract features from the processed images using a ResNet50-IBN-A model.

        Args:
            vector_collection (dict): A dictionary containing vector collections (features and metadata) extracted from the video.

        Returns:
            dict: The updated vector collection with extracted features.
        """
        curr_time = time.time()
        model = self._load_feature_extractor_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for i, (ky, val) in enumerate(vector_collection.items(), 1):
            image_data = val["image"]
            image_data = self._prepare_image(image_data)
            input_tensor = self.transform(image_data).unsqueeze(0).to(device)

            with torch.no_grad():
                extracted_features = model(input_tensor).cpu().numpy().flatten()

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
        logger.info("Supplied cameras: "+ str(videos))

        for i in videos:
            self.camera_names.append(i)  # os.path.basename(i).replace(".mp4", "")

        for i, video_path in enumerate(videos, 1):
            video_start_time = time.time()
            self.camera_name = os.path.basename(video_path).replace(".mp4", "")
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

            # Save results to HDF5
            save_results_start_time = time.time()
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            self.save_results(f"processed_results_{base_name}.hd5", vector_collection)
            save_results_time = time.time() - save_results_start_time
            logger.telemetry(
                f"Saved results for {self.camera_name} in {save_results_time:.2f} seconds"
            )
            self.experiment.log_metric(
                f"{self.camera_name}_save_results_time", save_results_time
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
            logger.critical("Working on cameras")
            self.check_video_stream(self.data_array[i], self.camera_names[i])

        for ky, val in self.data.items():
            for i in range(len(val)):
                self.data[ky][i][i] = 0

        logger.critical("Working on matrices")
        for ky, val in self.data.items():
            os.makedirs(f"{self.team_name}/Matrices", exist_ok=True)
            with open(f"{self.team_name}/Matrices/{ky}.json", "w") as f:
                json.dump(val, f)

        overall_process_time = time.time() - overall_start_time
        logger.telemetry(f"Overall processing time: {overall_process_time:.2f} seconds")

        self.experiment.log_metric("total_processing_time", overall_process_time)
        self.experiment.log_asset("application.log")
        self.experiment.log_asset("processed_results_*.hd5")

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
        return [i for i in data.values()]

    def process_results(self, results: List[VehicleMatch], source_camera: str) -> None:
        """
        Process the vehicle matching results and log the matches.

        Args:
            results (List[VehicleMatch]): List of vehicle matching results.
            source_camera (str): The name of the camera where the source video was recorded.
        """
        source_camera = os.path.basename(source_camera).removesuffix(".mp4")

        names = [os.path.basename(i).replace(".mp4", "") for i in self.camera_names]
        src_idx = names.index(source_camera)
        colors = [(255, 0, 0), (0, 0, 255)]

        for i in results:
            frames = self.get_frame_from_video(i["frame_id"], i["camera_name"])
            bounding_box = [
                int(float(k))
                for k in i.get("bounding_box", "")
                .split("[")[1]
                .split("]")[0]
                .split(",")
            ]
            if bounding_box:
                x1, y1, x2, y2 = map(int, bounding_box)
                cv2.rectangle(
                    frames, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=8
                )

                vehicle_name = self.process_vehicle_name_string(
                    self.team_name, i.get("vehicle", "unknown")
                )
                text = vehicle_name.split("/")[-1] + "_" + i.get("similar_vehicle_id")
                font_scale = 0.75
                font_thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_width, text_height = text_size
                text_position = (x1 + 10, y1 + 30)
                background_position = (
                    text_position[0],
                    text_position[1] - text_height,
                    text_position[0] + text_width,
                    text_position[1],
                )
                cv2.rectangle(
                    frames,
                    (background_position[0], background_position[1]),
                    (background_position[2], background_position[3]),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                color = colors[results.index(i) % len(colors)]
                cv2.putText(
                    frames,
                    text,
                    text_position,
                    font,
                    font_scale,
                    color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
                output_filename = f"{vehicle_name}_{i.get('camera_name').split('/')[-1].replace('.mp4', '')}_{i.get('vehicle_id')}.jpg"

                cv2.imwrite(output_filename, frames)
            self.data[i["vehicle"]][src_idx][
                names.index(i.get("camera_name").split("/")[-1].replace(".mp4", ""))
            ] += 1

    def process_vehicle_name_string(self, team_name: str, vehicle_name: str) -> str:
        """
        Process the vehicle name string and ensure directory structure is created.

        This method takes the vehicle name string (e.g., "Car-123") and splits it to get
        the vehicle type (e.g., "Car") and a unique identifier (e.g., "123"). It then ensures
        that a directory exists for the vehicle type and returns a formatted string with the
        directory structure included.

        Args:
            team_name (str): The name of the team, used to create the directory structure.
            vehicle_name (str): The vehicle name string containing the type and identifier.

        Returns:
            str: A formatted string with the directory structure, e.g., "Car/Car-123".
        """
        names = vehicle_name.split("#")
        vehicle_type = names[0].strip()

        directory_mapping = {
            "Car": "Car",
            "Bus": "Bus",
            "LCV": "LCV",
            "Truck": "Truck",
            "Three-Wheeler": "Three-Wheeler",
            "Two-Wheeler": "Two-Wheeler",
            "Bicycle": "Bicycle",
        }

        if vehicle_type in directory_mapping:
            directory_path = directory_mapping[vehicle_type]
            os.makedirs(f"{team_name}/Images/{directory_path}", exist_ok=True)
            return f"{team_name}/Images/{directory_path}/{vehicle_name}"

        return vehicle_name

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
        try :
            assert cap.isOpened(), "Error reading video file"
        except AssertionError as e:
            logger.error(e)
            raise

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in video: {total_frames}")

        self.experiment.log_metric("total_frames", total_frames)

        if self.max_frames == -1:
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
        while cap.isOpened() and frame_count < self.max_frames:
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
                logger.info(f"Processed {frame_count}/{self.max_frames} frames")

        logger.info(
            "Video processing completed in %.2f seconds.", time.time() - current_time
        )
        cap.release()

        self.experiment.log_metric("video_processing_time", time.time() - current_time)

        return counter.get_vector_collection()

    def save_photos(self, collection) -> None:
        """
        Save images from the vector collection to disk, including drawing bounding boxes and text.

        Args:
            collection (dict): The vector collection containing the images and metadata.
        """
        curr_time = time.time()
        colors = [(255, 0, 0), (0, 0, 255)]

        for i, (key, val) in enumerate(collection.items(), 1):
            image_data = val["image"]
            bounding_box = val.get("bounding_box")
            if bounding_box:
                x1, y1, x2, y2 = map(int, bounding_box)
                cv2.rectangle(
                    image_data, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=8
                )

                vehicle_name = self.process_vehicle_name_string(
                    self.team_name, val.get("vehicle", "unknown")
                )
                text = vehicle_name.split("/")[-1]
                font_scale = 0.75
                font_thickness = 1
                font = cv2.FONT_HERSHEY_SIMPLEX

                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_width, text_height = text_size
                text_position = (x1 + 10, y1 + 30)
                background_position = (
                    text_position[0],
                    text_position[1] - text_height,
                    text_position[0] + text_width,
                    text_position[1],
                )
                cv2.rectangle(
                    image_data,
                    (background_position[0], background_position[1]),
                    (background_position[2], background_position[3]),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                color = colors[i % len(colors)]
                cv2.putText(
                    image_data,
                    text,
                    text_position,
                    font,
                    font_scale,
                    color,
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )

            output_filename = f"{vehicle_name}_{key}.jpg"
            cv2.imwrite(output_filename, image_data)

            if i % 100 == 0:
                logger.info(f"Saved {i} photos with bounding boxes and text")

        logger.telemetry(f"Photos saved in {time.time() - curr_time:.2f} seconds")

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
                embeddings.append(data["features"].tolist())
                metadatas.append(
                    {
                        "camera": camera_name,
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
                ["chroma", "run", "--host", "0.0.0.0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
