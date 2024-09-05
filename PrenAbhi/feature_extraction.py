import numpy as np
import torch
import torchvision.transforms as transforms
import onnxruntime as ort


class BaseFeatureExtraction:
    def __init__(self, num_classes: int = 768, onnx_model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_classes = num_classes
        self.use_onnx = False

        if onnx_model_path:
            self.use_onnx = True
            self.onnx_model_path = onnx_model_path

    def process_bounding_box(self, img: torch.Tensor, bbox):
        if isinstance(bbox[0], list):
            bbox = bbox[0]
        x1, y1, x2, y2 = map(int, bbox)
        return img[:, y1:y2, x1:x2]

    def preprocess_image(self, cropped_img, size=224):
        preprocess = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return preprocess(cropped_img)

    def extract_features(self, img: np.ndarray, size=224):
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(self.device)
        input_tensor = (
            self.preprocess_image(img_tensor, size=size).unsqueeze(0).to(self.device)
        )

        if self.use_onnx:
            return self.run_onnx_inference(input_tensor)[0]
        else:
            with torch.no_grad():
                output = self.model(input_tensor)[0]
            return output

    def export_to_onnx(self):
        dummy_input = torch.randn(1, 3, 224, 224).to(
            self.device
        )  # Adjust size as necessary
        torch.onnx.export(
            self.model, dummy_input, self.onnx_model_path, opset_version=11
        )
        print(f"Model exported to {self.onnx_model_path}")

    def load_onnx_model(self):
        ort_session = ort.InferenceSession(
            self.onnx_model_path,
        )
        return ort_session

    def run_onnx_inference(self, input_tensor):
        if not hasattr(self, "ort_session"):
            self.ort_session = self.load_onnx_model()

        ort_inputs = {self.ort_session.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs


class EfficientNetFeatureExtraction(BaseFeatureExtraction):
    def __init__(self, num_classes: int = 768):
        super().__init__(num_classes)
        from torchvision import models

        self.model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )
        self.model.eval().to(self.device)


class MobileNetSmallFeatureExtraction(BaseFeatureExtraction):
    def __init__(self, num_classes: int = 512):
        super().__init__(num_classes)
        from torchvision import models

        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.model.classifier[3] = torch.nn.Linear(
            self.model.classifier[3].in_features, num_classes
        )
        self.model.eval().to(self.device)


class SqueezeNetFeatureExtraction(BaseFeatureExtraction):
    def __init__(self, num_classes: int = 768, onnx_model_path: str = None):
        super().__init__(num_classes, onnx_model_path)
        from torchvision import models

        self.model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
        self.model.classifier[1] = torch.nn.Conv2d(
            self.model.classifier[1].in_channels, num_classes, kernel_size=(1, 1)
        )
        self.model.eval().to(self.device)


class InceptionFeatureExtraction(BaseFeatureExtraction):
    def __init__(self, num_classes: int = 768):
        super().__init__(num_classes)
        from torchvision import models

        self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.model.eval().to(self.device)

    def extract_features(self, img: np.ndarray, bbox, size=299):
        return super().extract_features(img, bbox, size=size)
