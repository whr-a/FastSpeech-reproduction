# FastSpeech-Pytorch
The Implementation of FastSpeech Based on Pytorch.

## Prepare Dataset
1. Download and extract [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Put LJSpeech dataset in `FastSpeech/data`.
3. Unzip `alignments.zip`.
4. Put [Nvidia pretrained waveglow model](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing) in the `FastSpeech/waveglow/pretrained_model` and rename as `waveglow_256channels.pt`;
5. Run `python3 FastSpeech/preprocess.py`.

## Training
Run `python3 FastSpeech/train.py`.

## Evaluation
Run `python3 FastSpeech/eval.py`.

# Calculate WER 
The Implementation of using an open-source Automatic Speech Recognition (ASR) model instead of the traditional Mean Opinion Score (MOS) method.
## STT
Run `python3 STT.py`, and we can get a text.
## WER
Run `python3 wer.py`, and we can get the Word Error Rate (WER).