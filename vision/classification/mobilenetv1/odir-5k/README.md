#  MobileNetV1 (TF-Keras Model)

## Source
This model uses the pre-trained base model [MobileNet from](tf_keras.applications.MobileNet)
- **Backbone**: MobileNetV1 pretrained on ImageNet
- **Alpha**: `MBV1_ALPHA = 0.5`

## Environment
```
tf_keras: 2.19
cnn2snn: 2.17.0
quantizeml: 1.0.1
```

## References
- **MobileNets**: [Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- **Dataset**: [ODIR-5K Ocular disease recognition](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)

## License
Apache 2.0 License (models only)