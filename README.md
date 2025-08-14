# Akida Model Zoo

## Introduction

The **Akida Model Zoo** is an extension of our [foundation models](https://doc.brainchipinc.com/model_zoo_performance.html#akida-2-0-models), 
featuring a curated set of models validated on our Akida 2.0 FPGA platform. These models are
made available to support the broader adoption of the Akida solution by developers, researchers,
and enthusiasts alike.

## Models

Float and quantized models are available, with the quantized version converted and evaluated using
the Akida solution.

| Domain | Use case       | Architecture                                                   | Resolution | Dataset  | #Params | Quantization | Accuracy |
|--------|----------------|----------------------------------------------------------------|------------|----------|---------|--------------|----------|
| Vision | Classification | [MobileNetV2 0.75](vision/classification/mobilenetv2/imagenet) | 160        | ImageNet | 2.6M    | 8            | 62.85%   |
| Vision | Classification | [MobileNetV2 0.35](vision/classification/mobilenetv2/imagenet) | 96         | ImageNet | 1.2M    | 8            | 43.47%   |
| Vision | Classification | [MobileNetV2](vision/classification/mobilenetv2/CIFAR-10) | 128        | CIFAR-10 | 2.25M   | 8            | 93.96%   |
| Vision | Classification | [MobileNetV2](vision/classification/mobilenetv2/OXFORD_FLOWERS) | 224  | OXFORD_FLOWERS | 2.4M   | 8       | 91.97%   |
| Vision | Classification | [MobileNetV4](vision/classification/mobilenetv2/CIFAR-10) | 128        | CIFAR-10 | 2.5M   | 8            | 94.72%   |
| Vision | Classification | [MobileNetV4](vision/classification/mobilenetv2/OXFORD_FLOWERS) | 224  | OXFORD_FLOWERS | 2.6M   | 8       | 85.41%   |
| Vision | Classification | [Akida_MobileNet](vision/classification/mobilenetv2/CIFAR-10) | 128        | CIFAR-10 | 2.25M   | 8            | 91.92%   |
| Vision | Classification | [Akida_MobileNet](vision/classification/mobilenetv2/OXFORD_FLOWERS) | 224  | OXFORD_FLOWERS | 3.3M   | 8       | 91.08%   |


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
