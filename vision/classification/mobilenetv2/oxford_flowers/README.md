# MobileNetV2 (Custom Model)

## Source  
This model uses [**MobileNetV2**](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)
from TensorFlow Keras as the backbone, initialized with ImageNet pretrained weights.  
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

### Environment  
```
tensorflow: 2.15
keras: 2.15
akida_models: 1.7.0
quantizeml: 0.16.0
```


## Training Details  
- **Backbone**: MobileNetV2 pretrained on ImageNet  
- **Alpha**: `MBV2_ALPHA = 1`  


## References  
- **MobileNet-v2** Model from the paper [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  
- TensorFlow [MobileNetV2 documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2)  


## License  
Apache 2.0 License
