# MobileNetV2 (Custom Classification Model)

## Source  
This model uses [**MobileNetV2**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2) from TensorFlow Keras as the backbone, initialized with ImageNet pretrained weights.  
A rescaling layer `(1./127.5, offset=-1)` is applied for input normalization, and a custom dense classifier head is added.  

```python
def build_model():
    x = Input((IMG_SIZE, IMG_SIZE, 3))
    y = layers.Rescaling(1./127.5, offset=-1)(x)
    base = tf.keras.applications.MobileNetV2(
        input_tensor=y, alpha=MBV2_ALPHA, include_top=False, weights='imagenet',
        pooling='avg'
    )
    base.trainable = False
    z = layers.Dense(NUM_CLASSES, activation='softmax')(base.output)
    model = models.Model(x, z)
    return model, base
```

---

## Environment  
The following dependencies are required to build and run this model:  

```
tensorflow>=2.15
numpy
```

If exporting or quantizing for Akida hardware:  

```
onnx
onnxruntime
quantizeml
akida>=2.13.0
cnn2snn>=2.13.0
akida_models>=1.7.0
```

---

## Training Details  
- **Backbone**: MobileNetV2 pretrained on ImageNet  
- **Alpha**: Configurable via `MBV2_ALPHA`  


---

## References  
- MobileNetV2 Paper: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
- TensorFlow Docs: [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)  

---

## License  
Refer to the license of the original MobileNetV2 model:  
[https://huggingface.co/google/mobilenet_v2_0.75_160](https://huggingface.co/google/mobilenet_v2_0.75_160)  
