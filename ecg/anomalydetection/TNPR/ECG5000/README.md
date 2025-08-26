# TENNs Autoencoder

## Source
This model uses the TENNs autoencoder architecture (Tensorflow implementation of the Denoising application). 

The model is created when the train script is called sicne the create_model is integrated with the training script. Since it is an anomaly detection use case, the goal of the model is reconstructing the input signal unlike classification models wehere we will have to specify the number of classes.

## Environment
The following dependencies are required to build and run this model:

```bash
conda create -n tnpr python==3.10 -y
conda activate tnpr
```
```bash
pip install tensorflow[and-cuda]==2.15.*
pip install -r requirements.txt
```

If exporting or quantizing for Akida hardware:  

```
quantizeml
akida>=2.13.0
cnn2snn>=2.13.0
akida_models>=1.7.0
```


## References
 - aTENNuate - [aTENNuate: Optimized Real-time Speech Enhancement with Deep SSMs on Raw Audio](https://arxiv.org/html/2409.03377)