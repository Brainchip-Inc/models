# MobileNetV4-Conv-Small (Custom Classification Model)

## Source  
This model uses [**MobileNetV4-Conv-Small**](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k) from the **timm** library as the backbone, initialized with ImageNet pretrained weights.  
The classifier head is adapted for the target dataset by setting `num_classes = NUM_CLASSES`.  

```python
import timm

original_model = timm.create_model(
    "mobilenetv4_conv_small.e2400_r224_in1k",
    pretrained=True,
    num_classes=NUM_CLASSES
).eval()
```

---

## Environment  
The following dependencies are required to build and run this model:  

```
torch>=2.0
torchvision
timm
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

## Details  
- **Backbone**: MobileNetV4-Conv-Small pretrained on ImageNet  

---

## References  
- MobileNetV4 Paper: [Exploring the MobileNetV4 Architecture](https://arxiv.org/abs/2404.10518)  
- timm Model Card: [MobileNetV4-Conv-Small](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)  

---

## License  
Refer to the license of the original MobileNetV4 model:  
[https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k](https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k)  
