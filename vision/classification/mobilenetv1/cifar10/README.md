# MobileNetV1 (BrainChip Model)

## Source  
This model is an internal implementation of **MobileNetV1** for ImageNet classification,
provided through the BrainChip [akida-models](https://pypi.org/project/akida-models/)
python package.  
It supports both standard and quantized variants (4-bit and 8-bit) with adjustable width multiplier (`alpha`).  
See [online documentation](https://doc.brainchipinc.com/api_reference/akida_models_apis.html#akida_models.mobilenet_imagenet) for details

### Environment  
```
tensorflow: 2.15
keras: 2.15
akida_models: 1.7.0
quantizeml: 0.16.0
```

## References  
**MobileNet-v1** Model from the paper [Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)  


## License
Original model file: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py

The Keras original model file is licensed according to the notice below:

COPYRIGHT

Copyright (c) 2016 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.
The initial code of this repository came from https://github.com/keras-team/keras
(the Keras repository), hence, for author information regarding commits
that occured earlier than the first commit in the present repository,
please see the original Keras repository.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
