# Vehicle Re-Identification - The Bengaluru Mobility Challenge - Phase II, 2024

## Team Members

1. [Abhijith Ganesh](https://github.com/AbhijithGanesh)
2. [Prenitha Rajesh](https://github.com/PrenithaRajesh)
3. [Pardheev Krishna](https://github.com/PardheevKrishna)

This project is our submission to the [Bengaluru Mobility Challenge](https://ieee-dataport.org/competitions/bengaluru-mobility-challenge-2024) - Phase II. In this phase, the participants are tasked with re-identifying vehicles seen at some network locations at other locations of the network. The objective is to determine the origin-destination (O-D) flows for a specific part of the network over a defined time period. These O-D flow estimates are critical for transportation planning, what-if analysis, and related applications. This phase will conclude with finalist demos and the announcement of winners on **September 20, 2024**, in conjunction with the Symposium of Data for Public Good at IISc.

## Our approach

This approach basically utilizes the YOLOv5 object detection model to detect vehicles in the video frames. The detected vehicles are then passed through a feature extraction model to extract the features of the vehicles. These features are then compared to find similar vehicles in different frames. The similar vehicles are then matched to find the origin-destination flow of the vehicles.

We're extremely inspired from Prof Zhedong Zheng for his and his team's work on Re-ID and [AI City Challenge](https://www.aicitychallenge.org/) which initially motivated us to work on this project. [Connecting Language and Vision for Natural Language-Based Vehicle Retrieva](https://arxiv.org/pdf/2105.14897) which gave us the initial perspective on how to approach this problem.

## Project Setup

### 1. Pre-requisites

To run the project, ensure you have the following dependencies:

- Python 3.12
- PyTorch
- Nvidia GPU with CUDA cores for optimal performance

### 2. Installation

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### 3. Usage

#### VehicleMatch TypedDict

The `VehicleMatch` class is used for storing vehicle matching results. The attributes include:

- `vehicle_id` (str): Unique identifier for the vehicle.
- `similar_vehicle_id` (str): Identifier for a similar vehicle detected.
- `frame_id` (int): Frame number where the match was found.
- `boundingbox` (List[int]): Coordinates `[x1, y1, x2, y2]` representing the bounding box for the vehicle.

#### Solution Class

The `Solution` class handles video data processing and analysis, employing object detection and feature extraction using libraries like `YOLO`, `torch`, and `cv2`.

## Example Usage

To use the `Solution` class:

```python
from index import Solution

# Initialize the solution
solution = Solution()

# Process video data
solution.process_video("path/to/video.mp4")
```

## Documentation

### 1. Pre-requisites

- Python 3.12
- PyTorch
- Nvidia GPU with CUDA cores

### 2. Installation

Run the following command to install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Usage

#### VehicleMatch TypedDict

The `VehicleMatch` class is a typing class for vehicle matching results and includes the following attributes:

- `vehicle_id` (str): The unique identifier for the vehicle.
- `similar_vehicle_id` (str): The identifier for a similar vehicle found.
- `frame_id` (int): The frame number where the match was found.
- `boundingbox` (List[int]): The bounding box coordinates [x1, y1, x2, y2] for the detected vehicle.

#### Solution Class

The `Solution` class processes and analyzes video data using object detection and feature extraction with the help of libraries such as `YOLO`, `torch`, `cv2`, and more.

## Dependencies

The project uses the following libraries:

- `json`
- `logging`
- `os`
- `subprocess`
- `sys`
- `time`
- `chromadb`
- `cv2`
- `h5py`
- `numpy`
- `torch`
- `comet_ml`
- `PrenAbhi`
- `PIL`
- `torchvision`
- `ultralytics`

## Credits

- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [PyTorch](https://pytorch.org)
- [Comet.ml](https://www.comet.ml)
- [Ultralytics](docs.ultralytics.com)
- [Joint discriminative and generative learning for person re-identification](https://github.com/regob/vehicle_reid)
- [Dr Zhedong Zheng](https://www.zdzheng.xyz/)
