# Akida Model Zoo

## Introduction

The **Akida Model Zoo** expands our [foundation models](https://doc.brainchipinc.com/model_zoo_performance.html#akida-2-0-models)
with a hand-picked collection of models accelerated by the Akida 2.0 IP. Designed for developers,
researchers, and AI enthusiasts, these ready-to-use models make it easier than ever to explore,
build, and innovate with the Akida solution.

## Models

Both **float** and **quantized** models are available, with quantized versions converted and
evaluated on the **Akida solution**. For each model, the number of nodes required to run on a
minimal Akida IP configuration is provided, enabling straightforward assessment of performance
and deployment needs.

In addition, some models can be evaluated directly through [Akida Cloud](https://brainchip.com/aclp/)
☁️, offering a convenient way to explore and experiment without local hardware.

| Domain        | Use case          | Architecture                                                              | Resolution | Dataset       | #Params | Quantization | Accuracy | F1 Score | MSE   | Minimal #Nodes |
|---------------|-------------------|---------------------------------------------------------------------------|------------|---------------|---------|--------------|----------|----------|-------|----------------|
| Vision        | Classification    | [MobileNetV1_1.0](vision/classification/mobilenetv1/cifar10)              | 128        | CIFAR-10      | 2.25M   | 8            | 91.92%   |          |       | 5 ☁️          |
| Vision        | Classification    | [MobileNetV1_1.0](vision/classification/mobilenetv1/oxford_flowers)       | 224        | Oxford_Flower | 3.3M    | 8            | 91.08%   |          |       | 7              |
| Vision        | Classification    | [MobileNetV2 1.0](vision/classification/mobilenetv2/imagenet)             | 224        | ImageNet      | 3.5M    | 8            | 70.35%   |          |       | 7              |
| Vision        | Classification    | [MobileNetV2 0.75](vision/classification/mobilenetv2/imagenet)            | 160        | ImageNet      | 2.6M    | 8            | 62.85%   |          |       | 4 ☁️          |
| Vision        | Classification    | [MobileNetV2 0.35](vision/classification/mobilenetv2/imagenet)            | 96         | ImageNet      | 1.2M    | 8            | 43.47%   |          |       | 2 ☁️          |
| Vision        | Classification    | [MobileNetV4 1.0](vision/classification/mobilenetv4/imagenet)             | 224        | ImageNet      | 3.77M   | 8            | 71.86%   |          |       | 8              |
| Vision        | Classification    | [MobileNetV2_1.0](vision/classification/mobilenetv2/cifar10)              | 128        | CIFAR-10      | 2.25M   | 8            | 93.96%   |          |       | 5 ☁️          |
| Vision        | Classification    | [MobileNetV2_1.0](vision/classification/mobilenetv2/oxford_flowers)       | 224        | Oxford_Flower | 2.4M    | 8            | 91.97%   |          |       | 7              |
| Vision        | Classification    | [MobileNetV4_1.0](vision/classification/mobilenetv4/cifar10)              | 128        | CIFAR-10      | 2.5M    | 8            | 94.72%   |          |       | 7              |
| Vision        | Classification    | [MobileNetV4_1.0](vision/classification/mobilenetv4/oxford_flowers)       | 224        | Oxford_Flower | 2.6M    | 8            | 85.41%   |          |       | 8              |
| Vision        | Classification    | [spatiotemporal](vision/classification/spatiotemporal/FallVision)         | 224        | FallVision    | 1.34M   | 8            | 98.36%   |          |       | 16             |
| Vision        | Classification    | [MLP](vision/classification/MLP/MNIST)                                    | 784        | MNIST         | 203.5K  | 8            | 98.05%   |          |       | 1 ☁️          |
| Vision        | Classification    | [LogisticRegression](vision/classification/LogisticRegression/MNIST)      | 784        | MNIST         | 7.9K    | 8            | 84.52%   |          |       | 1 ☁️          |
|  ECG          | Classification    | [1DCNN](ecg/classification/1DCNN/MIT-BIH)                                 | 360        | MIT-BIH       | 74K     | 8            |          | 97.3%    |       | 1 ☁️          |
|  ECG          | Anomaly Detection | [1DCNN](ecg/anomalydetection/1DCNN/ECG5000)                               | 144        | ECG5000       | 290K    | 8            |          | 94.0%    |       | 1 ☁️          |
| Tabular       | Classification    | [LogisticR.](tabular/classification/LogisticRegression/Breast_Cancer)     | 30         | Breast_Cancer | 169     | 8            | 93.9%    |          |       | 1 ☁️          |
| Synthetic     | Regression        | [MLP](synthetic/regression/MLP/1D_Curve)                                  | 1          | 1D_Curve      | 6.2K    | 8            |          |          | 0.136 | 1 ☁️          |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/soda_bottles)| 224        | Soda_bottle   | 2.43M   | 8            | 91.53%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/trail_camera)| 224        | Trail_camera  | 2.43M   | 8            | 84.74%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/road_signs)  | 224        | Road_signs    | 2.43M   | 8            | 65.46%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/furniture)   | 224        | Furniture     | 2.43M   | 8            | 79.21%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/aerial-cows)        | 384        | Aerial_Cows          | 2.43M   | 8            | 30.91%   |          |       | 16         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/bees)               | 224        | Bees                 | 2.43M   | 8            | 59.99%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/cable-damage)       | 224        | Cable_Damage         | 2.43M   | 8            | 77.32%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/corrosion)          | 224        | Corrosion            | 2.43M   | 8            | 39.06%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/digits)             | 224        | Digits               | 2.43M   | 8            | 92.32%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/face_detection)     | 224        | Face_Detection       | 2.43M   | 8            | 75.98%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/hand-gestures)      | 224        | Hand_Gestures        | 2.43M   | 8            | 53.52%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/license_plate_detection)      | 224        | License_Plate        | 2.43M   | 8            | 96.22%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/liver-disease)      | 224        | Liver_Disease        | 2.43M   | 8            | 40.57%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/phages)             | 224        | Phages               | 2.43M   | 8            | 67.18%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/sign-language)      | 224        | Sign_Language        | 2.43M   | 8            | 85.88%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/stomata-cells)      | 224        | Stomata_Cells        | 2.43M   | 8            | 52.14%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/underwater-objects) | 224        | Underwater_Objects   | 2.43M   | 8            | 44.89%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/abs_obb)            | 384        | Ships_Detection      | 2.43M   | 8            | 39.60%   |          |       | 12         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/bone_fracture)      | 224        | Bone_Fracture        | 2.43M   | 8            | 60.70%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/facial_expression)  | 224        | Facial_Expression    | 2.43M   | 8            | 75.40%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/pothole_detection)  | 224        | Pothole_Detection    | 2.43M   | 8            | 57.20%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/search_and_rescue)  | 224        | Search_And_Rescue    | 2.43M   | 8            | 77.00%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/traffic_detection)  | 224        | Traffic_Detection    | 2.43M   | 8            | 71.80%   |          |       | 6 ☁️         |
| Vision        | Detection         | [AkidaNet18/CenterNet](vision/detection/akidanet18_centernet/weed_crop)          | 384        | Weed_Crop            | 2.43M   | 8            | 47.70%   |          |       | 16          |

## Download
### Git Clone
To avoid downloading the models during cloning due to their large size:
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Brainchip-Inc/models.git
```

To download a specific model:
```bash
git lfs pull --include="[path to model]" --exclude=""
```

To download all models:
```bash
git lfs pull --include="*" --exclude=""
```

### GitHub UI
Alternatively, you can download models directly from GitHub. Navigate to the model's page and
click the "Download" button on the top right corner.

## Model Visualization

For a graphical representation of each model's architecture, we recommend using [Netron](https://netron.app/).
