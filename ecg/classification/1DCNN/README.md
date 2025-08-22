
# ECG Arrhythmia Classification with 2D CNN and Quantization

This project performs ECG heartbeat classification (4 classes) using a compact 2D CNN on the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/). The model is then post-training quantized and converted to an Akida-compatible spiking neural network (SNN), followed by final evaluation of the SNN model.
Classifying irregular heartbeats is critical for identifying their origin and assessing the potential severity of arrhythmias, which directly impacts diagnosis and treatment. Key discriminative features lie in **QRS morphology and timing** of the beats, making **CNNs**  well-suited for this task due to their ability to efficiently capture local temporal patterns.


## Source  
The internal code repository for this project: [**Repository**](https://git.corp.brainchipinc.com/brainchipResearch/akida_models_bootcamp/)


---

## Environment  
The following dependencies are required to build and run this model:  

```
matplotlib==3.8.4
numpy==1.26.4
pandas==2.3.0
scikit-learn==1.7.0
scipy==1.16.0
tensorflow==2.15.0
wfdb==4.3.0
cnn2snn==2.13.0
quantizeml==0.16.0
```



## Run

```bash
pip install -r requirements.txt
python main.py          # train & evaluate, quantize, test, and export


