**Status:**: Unfinished (code is provided as-is, to be run it needs some manipulations, can be updated if I find time)

# MT-CNN for EEG Emotion Recognition
Code for the paper "Multi-Task CNN Model for Emotion Recognition from EEG Brain Maps" presented on the 4th IEEE International Conference on Bio-engineering for Smart Technologies BioSmart 2021. Keras implentation.

It describes a simple model which outperforms previous SOTA approaches in terms of accuracy on the DEAP dataset.

[Slides](https://github.com/dolphin-in-a-coma/multi-task-cnn-eeg-emotion/blob/main/MT_CNN.pdf) 
[Video]
[Colab]
Paper (will be uploaded on the IEEE Xplore in next few months) 

## Pros and Cons

As a subject-dependent approach, the solution nevertheless provides the framework for a unified fully 2D-CNN model for solving two tasks on the whole set of subjects without regarding the number of subjects. This approach previously led to the performance degradation because of high cross-subject variance in the data, thus in previous state-of-the-art researches the common practice was to use unique model for each task and for each subject. This problem was overcome by using a set of regularizations and best training practices.

Nevertheless, our model cannot be used for subject-independent emotion recognition, specifically if EEG signals were recorded using other EEG sensor types. When this model was used to test on subjects which weren't seen during the training step, the accuracy fell down below 60%. I didn't see in the literature any high-accurate approaches for subject-independent emotion recognition, so it's still an interesting and highly challenging task.

## Results
| Method | DEAP Valence Accuracy | DEAP Arousal Accuracy|
| ----------- | ----------- |----------- |
| [Multi-column CNN](https://www.mdpi.com/1424-8220/19/21/4736) | 90.01% | 90.65%|
| [SAE-LSTM](https://www.frontiersin.org/articles/10.3389/fnbot.2019.00037/full) | 81.10% | 74.38%|
| [4D-CRNN](https://www.researchgate.net/publication/344371728_EEG-based_emotion_recognition_using_4D_convolutional_recurrent_neural_network) | 94.22% | 94.58%|
| [FBCCNN](https://www.hindawi.com/journals/cmmm/2021/2520394/) | 90.26% | 88.90%|
| MT-CNN (Proposed method) | 96.28% | 96.62%|

## Dataset
[DEAP dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) consists of electroencephalography signals from recorded 32 participants. During the sessions participants watched 40 music videoclips and marked them in terms in terms of valence, arousal, liking and dominance from 1 to 9. In this research we use only arousal and valence values, which was split in low and high regions to solve a classification task.


## Data preprocessing


## Training 
