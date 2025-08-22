# Autism detection in Children (hand-built model)

## Source  
This model is an implementation of custom CNN model on the Kaggle autism dataset containing images of children https://www.kaggle.com/datasets/cihan063/autism-image-data/data 


### Environment  
```
akida ~= 2.14.0
cnn2snn ~= 2.14.0
akida_models ~= 1.8.0
```

## References  
https://pmc.ncbi.nlm.nih.gov/articles/PMC12283938/ Automated identification of autism spectrum disorder from facial images using explainable deep learning models report on various accuracies for autism detection on the same Kaggle autism dataset https://www.kaggle.com/datasets/cihan063/autism-image-data/data as follows:
```
MobileNet        - 88%
Xception         - 87.70%
Inception V3     - 86.10%
EfficientNetB0   - 85.60%
EfficientNetB7   - 82.60%
VGG16            - 86.30%
```
whereas Hand-built model gives Torch Test accuracy: 84% (close to the highest range as opposed to heavy size models listed above)
- Other metrics of the model
```
Model size (onnx): 2.13 MB
Quantized model size (onnx): 650.05 kB
Onnx Floating point model accuracy: 84%
Quantized model accuracy: 84%
Akida conversion accuracy: 82.67% 
```
