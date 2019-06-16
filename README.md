# UjipenChars2 handwritten letters classifier with Gated Recurrent Unit (GRU)

This repository supplements [stm32f429-chars](https://github.com/dizcza/stm32f429-chars) repository to train a recurrent neural network that will be used later on in a microcontroller.

A small list of manually picked examples from train data which confuse classifiers is put in [`dropped.txt`](ujipenchars2/dropped.txt). All test samples from [UjipenChars2](https://archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters+(Version+2)) dataset are used during the model validation.

The main file is [`gru.py`](gru.py), where the training procedure of GRU is defined alongside with the test (validation) score.

Initially started with DTW as a baseline algorithm to find the closest pattern from the train data, given an input sample. DTW-related implementation is moved to [dtw](https://github.com/dizcza/ujipen/tree/dtw) branch.

To give you the rough approximation of performance of both classifiers,

|                     |  GRU   |  DTW   |
|---------------------|--------|--------|
| Validation accuracy | 98.3 % | 81.9 % |

But the main difference between those two is their inference time: GRU is much faster than DTW due to parallel computation.
