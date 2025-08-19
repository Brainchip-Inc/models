# MobileNetV2

## Source
Float models are from HuggingFace and were retrieved through the optimum tool
```
optimum-cli export onnx --model google/mobilenet_v2_1.0_224 mobilenet_v2_1.0_224
optimum-cli export onnx --model google/mobilenet_v2_0.75_160 mobilenet_v2_0.75_160
optimum-cli export onnx --model google/mobilenet_v2_0.35_96 mobilenet_v2_0.35_96
```
### Environment
```
onnx: 1.16.1
onnxruntime: 1.19.0
optimum: 1.23.3
quantizeml: 0.12.1
```

## References
**MobileNet-v2** Model from the paper [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

## License
Please refer to https://huggingface.co/google/mobilenet_v2_1.0_224
