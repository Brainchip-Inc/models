# ECG Anomaly Detection with Quantized Autoencoder

This project implements an anomaly detector for ECG data using a convolutional autoencoder, evaluated in both float and quantized (8-bit) form and converted to SNN.


## Source  
The internal code repository for this project: [**Repository**](https://git.corp.brainchipinc.com/brainchipResearch/akida_models_bootcamp/tree/project_anomaly_AE)




---

## Data & Preprocessing

- **Dataset:** A version of [ECG5000](https://timeseriesclassification.com/description.php?Dataset=ECG5000) that is alraeady preprocessed and convered to binay labels for normal and anomaly, loaded via  
  `http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv`
- **Labels:**  
    - `1` = Normal beat  
    - `0` = Anomaly  
- **Preprocessing:**  
    - Normalize all beats globally to [0, 1] 
    - Pad to length 144, reshape to (12, 12, 1) for 2D CNN  
    - Convert to `uint8` for quantized inference

**ECG5000** is a standard arrhythmia dataset with 5,000 labeled heartbeat segments (1D, length 140), used for anomaly detection and ECG classification benchmarks.

---
## Environment  
The following dependencies are required to build and run this model:  

```

matplotlib==3.8.4
seaborn==0.13.2
numpy==1.26.4
pandas==2.3.0
scikit-learn==1.7.0
tensorflow==2.15.0
cnn2snn==2.13.0
quantizeml==0.16.0

---

## Run

```bash
pip install -r requirements.txt
python main.py          # train & evaluate, quantize, test, and export


