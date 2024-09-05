import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from typing import List, Dict, Tuple, Union


def reidentify_vehicles(
    vector_db_cam1, vector_db_cam2, threshold: float = 0.8
) -> Tuple[Dict[str, np.ndarray], List[Tuple[str, int, str, int]]]:
    class_matrices: Dict[str, np.ndarray] = {
        cls: np.zeros((2, 2))
        for cls in [
            "Bicycle",
            "Bus",
            "Car",
            "LCV",
            "Three-Wheeler",
            "Truck",
            "Two-Wheeler",
        ]
    }
    matches = []

    for id1, data1 in vector_db_cam1.items():
        features1: np.ndarray = data1["features"]
        class_name1: str = data1["class_name"]

        for id2, data2 in vector_db_cam2.items():
            if data1["class_name"] != data2["class_name"]:
                continue

            features2: np.ndarray = data2["features"]
            similarity: float = cosine_similarity(
                features1.reshape(1, -1), features2.reshape(1, -1)
            )[0][0]

            if similarity > threshold:
                class_matrices[class_name1][0][1] += 1
                matches.append((id1, data1["timestamp"], id2, data2["timestamp"]))

    return class_matrices, matches
