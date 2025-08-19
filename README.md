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


| Domain | Use case       | Architecture                                                      | Resolution | Dataset  | #Params | Quantization | Accuracy | Minimal #Nodes |
|--------|----------------|-------------------------------------------------------------------|------------|----------|---------|--------------|----------|----------------|
| Vision | Classification | [MobileNetV2 1.0](vision/classification/mobilenetv2/imagenet)     | 224        | ImageNet | 3.5M    | 8            | 70.35%   | 7              |
| Vision | Classification | [MobileNetV2 0.75](vision/classification/mobilenetv2/imagenet) ☁️  | 160        | ImageNet | 2.6M    | 8            | 62.85%   | 4              |
| Vision | Classification | [MobileNetV2 0.35](vision/classification/mobilenetv2/imagenet) ☁️  | 96         | ImageNet | 1.2M    | 8            | 43.47%   | 2              |

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
