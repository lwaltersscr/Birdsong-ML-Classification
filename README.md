# Birdsong-ML-Classification
Identifying bird species from audio spectrogram using machine learning.


This project explores how machine learning can be used to automatically identify bird species from audio recordings. The motivation comes from biodiversity monitoring: birds are strong indicators of ecosystem health, but traditional surveys rely on human observers and are difficult to scale. By using audio recordings and machine learning, it becomes possible to monitor bird populations more efficiently and over larger areas.

##Overview

The goal of the project is to train a model that takes a short audio clip of a bird call and predicts which species produced it. I use the BirdCLEF 2023 dataset, which contains thousands of labeled bird recordings collected from Xeno-Canto. Each recording is associated with metadata such as the species label and recording location.

To make the task manageable, I focus on the top 20 most common bird species in the dataset and frame the problem as a multi-class classification task.

##Data

The dataset consists of:

Audio files (.ogg) containing bird calls

Metadata (train_metadata.csv) that links each audio file to a bird species label

The raw audio is downsampled to 32 kHz and standardized to 5-second clips. Since different species appear at very different frequencies, the data is converted into mel spectrograms, which represent how sound energy changes across frequencies over time.

##Method

The main steps of the pipeline are:

Load and preprocess audio clips

Convert audio into mel spectrograms

Apply data augmentation (time and frequency masking) during training

Train a convolutional neural network (CNN) to classify species

To address class imbalance, I use class-weighted cross-entropy loss so that rarer species are not ignored during training. The model is trained and evaluated using a stratified train/validation split.

##Model

The model is a simple CNN that treats spectrograms as images. It consists of several convolutional layers followed by pooling and a final fully connected layer that outputs class probabilities. The model is trained from scratch using the Adam optimizer.

##Results

After five epochs of training, both training and validation loss decrease steadily, while accuracy increases well above random chance for a 20-class problem. Although the model is not fully converged, the results show that the network is learning meaningful acoustic patterns from the data. With additional training time or pretrained audio models, performance would likely improve further.

##What I Learned

This project highlighted how important preprocessing and data decisions are for machine learning, especially with audio. I learned how to work with spectrograms, handle class imbalance, and interpret training behavior over time. It also reinforced that early training results can look weak even when the overall approach is sound.

##Future Work

Possible next steps include training for more epochs, using pretrained audio models, incorporating longer audio clips with windowed inference, or using additional metadata such as geographic location to improve predictions.

