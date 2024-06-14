# TSE_PI
Official Code for Target Sound Extraction under Reverberant Environments with Pitch Information (Interspeech 2024)

## Introduction

### Stage 1 Conditional Pitch Estimation (F0 Extraction)


1. Dataset: selected from FSD50K 

2. RIR Simulation: Image-Source Method

    Reference link: https://www.audiolabs-erlangen.de/fau/professor/habets/software/smir-generator


3. Pitch Label: Generate by Praat under anechoic condition and labeled with time shift

    Reference link: https://parselmouth.readthedocs.io/_/downloads/en/stable/pdf/

4. TODO: Update the dataset.py (Update Soon!)


5. Command: python train_f0.py

### Stage 2 Target Sound Extraction with Pitch Information

Thanks for the open-source code from Veluri et al. for providing [Waveformer](https://github.com/vb000/Waveformer/) and [SemanticHearing](https://github.com/vb000/SemanticHearing).

The majority for this part is to simply add the pitch information and train similarly to the Waveformer.

## Reference Code

1. https://github.com/vb000/Waveformer/

2. https://github.com/vb000/SemanticHearing

3. https://github.com/lihan941002/Param-GTFB-GCFB

