<a href="https://colab.research.google.com/github/artmen1516/emotions-classification-seedv/blob/main/process_seedv_raw_files_with_mne_from_drive.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# EEG-Based Emotion Classification Using Deep Learning Models

## Overview

This repository contains the code and resources for a research project focused on classifying emotional states using EEG signals. The study explores the performance of three advanced deep learning models: **ShallowFBCSPNet**, **Deep4Net**, and **EEGNetv4**. These models are specifically designed to process EEG data and classify emotions such as happiness, sadness, disgust, neutrality, and fear.

## Features

- Implementation of three neural network architectures optimized for EEG data.
- Preprocessing of raw EEG signals, including filtering, segmentation, and feature extraction.
- Training and evaluation of models using the SEED-V dataset.
- Analysis of model performance with visualizations such as confusion matrices and loss curves.

## Getting Started

### Prerequisites

To run the code in this repository, you will need the following software and libraries:

- **Python 3.10 or higher**
- **Google Colab** (recommended for running with GPU support)
- **MNE-Python** for EEG signal processing
- **PyTorch** for implementing and training deep learning models
- **Matplotlib** for visualizing results
- **NumPy** and **Pandas** for data manipulation
