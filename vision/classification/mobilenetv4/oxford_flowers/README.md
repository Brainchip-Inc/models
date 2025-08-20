# MobileNetV4-Conv-Small (Custom Model)

## Source  
This model uses [**MobileNetV4-Conv-Small**](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)
from the **timm** library as the backbone, initialized with ImageNet pretrained weights.  
The classifier head is adapted for the target dataset by setting `num_classes = NUM_CLASSES`.  

```python
import timm

original_model = timm.create_model(
    "mobilenetv4_conv_small.e2400_r224_in1k",
    pretrained=True,
    num_classes=NUM_CLASSES
).eval()
```

### Environment  
The following dependencies are required to build and run this model:  

```
torch: 2.0
torchvision
timm
quantizeml: 0.16.0
```


## Training Details  
- **Backbone**: MobileNetV4-Conv-Small pretrained on ImageNet  


## References  
- **MobileNet-v4** Model from the paper [Exploring the MobileNetV4 Architecture](https://arxiv.org/abs/2404.10518)  
- timm Model Card [MobileNetV4-Conv-Small](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)  


## License  
Please refer to [https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)  
