# MobileNetV1 (BrainChip Internal ImageNet Model)

## Source  
This model is an internal implementation of **MobileNetV1** for ImageNet classification, provided in the BrainChip `akida_models` repository.  
It supports both standard and quantized variants (4-bit and 8-bit) with adjustable width multiplier (`alpha`).  

Internal repo: [akida_models/imagenet](https://git.corp.brainchipinc.com/brainchipResearch/akida_models_fork/tree/main/akida_models/imagenet)  

```python
from akida_models.imagenet import mobilenet_imagenet

model = mobilenet_imagenet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=1.0,
    include_top=True,
    classes=NUM_CLASSES,
    use_stride2=True,
    input_scaling=None
)
```

---

## Environment  
Required dependencies:  

```
tensorflow>=2.15
keras
numpy
```

(Other utilities such as `utils.imagenet_utils`, `layer_blocks`, and `model_io` are included in the internal BrainChip repo.)  

---

## Details  
- **Backbone**: MobileNetV1   
- **Width multiplier (`alpha`)**: 1.0 / 0.5 / 0.25  
---

## References  
- MobileNetV1 Paper: [Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  

---
