import os

from comet_ml import Experiment
from ultralytics import YOLO

experiment = Experiment(
    api_key=os.environ.get("COMET_API_KEY"), project_name="comet-ml-yolov-ag"
)

model = YOLO("yolov10x.pt")

total_epochs = 100
batch_size = 16
data_yaml = "datasets/clean.yaml"

experiment.log_code()

experiment.log_parameters(
    {
        "model": "yolov10x-v1",
        "epochs": total_epochs,
        "batch_size": batch_size,
    }
)

train_results = model.train(data=data_yaml, epochs=total_epochs, batch=batch_size)

experiment.log_metrics(train_results)

val_results = model.val(data=data_yaml, split="val")
experiment.log_metrics(val_results, prefix="val_")

test_results = model.val(data=data_yaml, split="test")
experiment.log_metrics(test_results, prefix="test_")

model.export(format="engine")  # Export as TensorRT engine
experiment.log_model("final_model", "yolov10l_final.engine")

runs_dir = "runs"
if os.path.exists(runs_dir):
    experiment.log_asset_folder(runs_dir, log_file_name=True)

experiment.end()
