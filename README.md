# Akida Model Zoo

## Introduction

The **Akida Model Zoo** is an extension of our [foundation models](https://doc.brainchipinc.com/model_zoo_performance.html#akida-2-0-models), 
featuring a curated set of models validated on our Akida 2.0 FPGA platform. These models are
made available to support the broader adoption of the Akida solution by developers, researchers,
and enthusiasts alike.

## Models

_Work in progress_


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
