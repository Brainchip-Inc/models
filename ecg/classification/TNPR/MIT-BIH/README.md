# TENNs classifier

## Source
This model is similar to the TENNs model used in the sc10 project.
We create a model aligned to our use case using the create_model.py file.
python create_model.py 

```bash
python create_model.py -s ./models/ecg_classifier_v5_1ch_float.h5 -i 360 1 -c 4 -ds 2 2 2 1  
```
where i is the input shape of the signal in the dataset, c is the numebr of classes, ds is the number of downsamplings and dowensampling factor for each.

You can utilize the code snippets present in the train_ech.sh for more information on training, evaluation and quantization.

## Environment

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
 - PLEIADES: [Building Temporal Kernels with Orthogonal Polynomials](https://arxiv.org/pdf/2405.12179)