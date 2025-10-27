# MobileNetV2 (Custom Model)

## Source
This model uses [**MobileNetV2**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
from TensorFlow Keras as the backbone, initialized with ImageNet pretrained weights.
- **Backbone**: MobileNetV2 pretrained on ImageNet
- **Alpha**: `MBV2_ALPHA = 1`

## Environment
```
tensorflow: 2.15
keras: 2.15
akida_models: 1.7.0
quantizeml: 0.16.0
```

## References
- **MobileNet-v2** Model from the paper [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- TensorFlow [MobileNetV2 documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)

## License
Apache 2.0 License
