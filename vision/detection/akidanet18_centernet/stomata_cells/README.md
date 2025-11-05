#  AkidaNet18/CenterNet (Custom Model)

## Source
This model uses the pre-trained base model centernet_akidanet18_voc_224.h5 from our data center: [**centernet_akidanet18_voc_224**](https://data.brainchip.com/models/AkidaV2/centernet/centernet_akidanet18_voc_384.h5)
- **Backbone**: AkidaNet18 backbone
- **Head**: CenterNet

## Environment
```
tf_keras: 2.19
cnn2snn: 2.17.0
quantizeml: 1.0.1
```

## References
- **AkidaNet18**: Akida model API [AkidaNet18 architecture](https://doc.brainchipinc.com/api_reference/akida_models_apis.html#akida_models.akidanet18_imagenet)
- **Dataset**:  [stomata cells detection](https://universe.roboflow.com/roboflow-100/stomata-cells/dataset/2)

## License
Apache 2.0 License (models only)
