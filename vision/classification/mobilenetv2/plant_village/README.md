# MobileNetV2 0.5 — Plant Village

## Source
- **Backbone**: [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) (tf.keras), ImageNet pretrained
- **Alpha**: `MBV2_ALPHA = 0.5`
- **Input resolution**: 224 × 224 × 3
- **Classes**: 38 (Plant Village)
- **Quantization**: 8-bit weights / 8-bit activations (QAT, `_i8_w8_a8.h5`)

## Performance

| Variant               | Top-1 accuracy |
|-----------------------|----------------|
| Float                 | 99.47%         |
| Quantized 8w8a8 (QAT) | 99.12%         |

Quantized-model sparsity — ReLU: 89%, overall: 53%.

## Environment

Install the exact versions used to export and convert these checkpoints — older
or newer versions may fail to round-trip the `quantizeml` custom layers or produce
a different Akida layer graph:

```bash
pip install \
    akida==2.17.0 \
    cnn2snn==2.17.0 \
    quantizeml==1.0.1
```

## Files

- `mobilenet_v2_0.5_plant_village.h5` — float Keras model
- `mobilenet_v2_0.5_plant_village_i8_w8_a8.h5` — QAT 8w8a8 Keras model

## Load the quantized model and convert to Akida

```python
import quantizeml
from cnn2snn import convert, set_akida_version, AkidaVersion

keras_model = quantizeml.load_model("mobilenet_v2_0.5_plant_village_i8_w8_a8.h5")

with set_akida_version(AkidaVersion.v2):
    akida_model = convert(keras_model)

akida_model.summary()
# Input shape:  [224, 224, 3]
# Output shape: [1, 1, 38]
```

## Map on device (optional)

```python
import akida

device = akida.SixNodesIPv2()       # or akida.devices()[0] for an Akida 2.0 FPGA
akida_model.map(device, hw_only=True)
akida_model.summary()
```

## References
- MobileNetV2 — [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- Dataset — [PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village)
- TensorFlow [MobileNetV2 docs](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)

## License
Apache 2.0 License (models only)
