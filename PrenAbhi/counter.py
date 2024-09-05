# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
from ksuid import Ksuid
from torchvision import transforms
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(
        self,
        names,
        reg_pts=None,
        count_reg_color=(255, 0, 255),
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,
    ):
        # Initialization code
        self.names = names
        self.reg_pts = reg_pts or [(20, 400), (1260, 400)]
        self.line_dist_thresh = line_dist_thresh
        self.counting_region = self._initialize_counting_region()
        self.region_color = count_reg_color
        self.region_thickness = region_thickness

        self.im0 = None
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts

        self.annotator = None
        self.window_name = "Ultralytics YOLOv8 Object Counter"

        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = {}
        self.count_txt_thickness = 0
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.cls_txtdisplay_gap = cls_txtdisplay_gap
        self.fontsize = 0.6

        self.track_history = defaultdict(list)
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks
        self.track_color = track_color

        self.track_id_to_ksuid = {}
        self.vector_collection = {}

        self.env_check = check_imshow(warn=True)

    def _initialize_counting_region(self):
        if len(self.reg_pts) == 2:
            return LineString(self.reg_pts)
        elif len(self.reg_pts) >= 3:
            return Polygon(self.reg_pts)
        else:
            return LineString([(20, 400), (1260, 400)])

    def _return_ndarray(self, bbox) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        return self.im0[y1:y2, x1:x2]

    def _return_tensor(self, bbox, im0) -> torch.Tensor:
        x1, y1, x2, y2 = map(int, bbox)
        image_np = im0[y1:y2, x1:x2]
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        preprocess = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return preprocess(image_tensor / 255.0)

    def _process_track(self, box, track_id, cls, frame_id):
        key = f"{self.names[cls]}#{track_id}"
        if key not in self.track_id_to_ksuid:
            self.track_id_to_ksuid[key] = str(Ksuid())

        self.vector_collection[self.track_id_to_ksuid[key]] = {
            "image": self.im0,
            "bounding_box": box,
            "frame_id": frame_id,
            "vehicle": key,
        }

        if self.names[cls] not in self.class_wise_count:
            self.class_wise_count[self.names[cls]] = {"IN": 0, "OUT": 0}

        if track_id not in self.track_history:
            self.track_history[track_id] = []

        track_line = self.track_history[track_id]
        track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
        if len(track_line) > 30:
            track_line.pop(0)

        return track_line

    def _update_count(self, track_line, box, track_id, cls):
        prev_position = (
            self.track_history[track_id][-2]
            if len(self.track_history[track_id]) > 1
            else None
        )

        if len(self.reg_pts) >= 3:
            is_inside = self.counting_region.contains(Point(track_line[-1]))
            if prev_position and is_inside and track_id not in self.count_ids:
                self.count_ids.append(track_id)
                if (box[0] - prev_position[0]) * (
                    self.counting_region.centroid.x - prev_position[0]
                ) > 0:
                    self.in_counts += 1
                    self.class_wise_count[self.names[cls]]["IN"] += 1
                else:
                    self.out_counts += 1
                    self.class_wise_count[self.names[cls]]["OUT"] += 1

        elif (
            len(self.reg_pts) == 2 and prev_position and track_id not in self.count_ids
        ):
            distance = Point(track_line[-1]).distance(self.counting_region)
            if distance < self.line_dist_thresh:
                self.count_ids.append(track_id)
                if (box[0] - prev_position[0]) * (
                    self.counting_region.centroid.x - prev_position[0]
                ) > 0:
                    self.in_counts += 1
                    self.class_wise_count[self.names[cls]]["IN"] += 1
                else:
                    self.out_counts += 1
                    self.class_wise_count[self.names[cls]]["OUT"] += 1

    def _generate_labels_dict(self):
        labels_dict = {}
        for key, value in self.class_wise_count.items():
            if value["IN"] or value["OUT"]:
                if not self.view_in_counts and not self.view_out_counts:
                    continue
                elif not self.view_in_counts:
                    labels_dict[key.capitalize()] = f"OUT {value['OUT']}"
                elif not self.view_out_counts:
                    labels_dict[key.capitalize()] = f"IN {value['IN']}"
                else:
                    labels_dict[key.capitalize()] = (
                        f"IN {value['IN']} OUT {value['OUT']}"
                    )
        return labels_dict

    def extract_and_process_tracks(self, tracks, frame_id):
        """Extracts and processes tracks for object counting in a video stream."""
        self.annotator = Annotator(self.im0, self.tf, self.names)

        if tracks and tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu().tolist()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, clss):
                track_line = self._process_track(box, track_id, cls, frame_id)
                self._update_count(track_line, box, track_id, cls)

    def display_frames(self):
        """Displays the current frame with annotations and regions in a window."""
        if self.env_check:
            cv2.namedWindow(self.window_name)
            if len(self.reg_pts) == 4:
                cv2.setMouseCallback(
                    self.window_name,
                    self.mouse_event_for_region,
                    {"region_points": self.reg_pts},
                )
            cv2.imshow(self.window_name, self.im0)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def save_results(self, filename: str):
        with h5py.File(filename, "w") as f:
            for key, value in self.vector_collection.items():
                group = f.create_group(key)
                group.create_dataset("image", data=np.array(value["image"]))
                group.attrs["frame_id"] = value["frame_id"]
                group.attrs["vehicle"] = value["vehicle"]

    def start_counting(self, im0, tracks, frame_id):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
            frame_id (int): Current frame ID.
        """
        self.im0 = im0
        self.extract_and_process_tracks(tracks, frame_id)
        if self.view_img:
            self.display_frames()
        return self.im0

    def get_vector_collection(self):
        return self.vector_collection
