"""
Loading ChuongDuong matlab dataset
"""

# Import required libraries
# %%
# create data use DWT and fuzz c-mean
# logic matrix sau khi qua tiền xử lý là DWT sẽ được tính fuzz c-mean
# sau đó lưu lại data set với lượng feature mới
# import pywt
import matplotlib.pyplot as plt
import numpy as np
# from tslearn.generators import random_walks
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance
# from tslearn.piecewise import PiecewiseAggregateApproximation
# from tslearn.piecewise import SymbolicAggregateApproximation, \
#     OneD_SymbolicAggregateApproximation
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# from tensorflow.keras.layers import *
# from tensorflow import keras
# import tensorflow as tf
from sklearn.metrics import confusion_matrix
# from keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Flatten, ReLU
#gpu = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu[1], True)
import pandas as pd
from IPython.display import display, HTML

import os
from scipy.io import loadmat
# from scipy.interpolate import interp1d

import numpy as np
import random


class Data_process:
    def data_loader(path):

        directory = path

        all_data = {}

        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.mat'):
                filepath = os.path.join(directory, filename)
                # Load the .mat file and add its contents to the dictionary
                mat_data = loadmat(filepath)
                
                # Use filename (without extension) as key for the data
                key = os.path.splitext(filename)[0]
                all_data[key] = mat_data['acceleration']
                # print(filepath)
                # print(mat_data)
        

        keys_to_stack = [f'ChuongDuong{i}' for i in range(11)]
        input_data = np.stack([all_data[key] for key in keys_to_stack], axis=0)
        # print(all_data.keys)

        # Create the corresponding labels
        output_labels = np.linspace(0,10,11)  # Using 0 and 1 as class labels for binary cross-entropy

        input_data = input_data[:,:,:40000]
        input_data.shape, output_labels.shape

        
        return input_data, output_labels


    def augment_time_series_data(input_data, labels, num_augmentations=5):
        """
        Augment time series data.

        :param input_data: Original time series data array.
        :param labels: Corresponding labels for the data.
        :param num_augmentations: Number of augmented samples to generate per original sample.

        :return: Augmented data array and corresponding labels.
        """
        augmented_data = []
        augmented_labels = []

        num_samples, num_channels, sequence_length = input_data.shape
        #print (sequence_length)

        for i in range(num_samples):
            for _ in range(num_augmentations):
                # Choose a random augmentation technique
                augmentation_type = random.choice(['noise', 'reverse', 'crop_pad'])

                if augmentation_type == 'noise':
                    # Add random noise
                    noise = np.random.normal(0, 0.001, input_data[i].shape)
                    augmented_sample = input_data[i] + noise

                elif augmentation_type == 'reverse':
                    # Reverse the sequence
                    augmented_sample = np.flip(input_data[i], axis=-1)

                elif augmentation_type == 'crop_pad':
                    # Crop and pad the sequence
                    crop_size = random.randint(0, sequence_length // 100)
                    padded_sample = np.pad(input_data[i], ((0, 0), (crop_size, 0)), mode='constant', constant_values=0)
                    augmented_sample = padded_sample[:, :-crop_size]

                augmented_data.append(augmented_sample)
                augmented_labels.append(labels[i])

        # Convert to numpy arrays
        augmented_data = np.array(augmented_data)
        augmented_labels = np.array(augmented_labels)

        return augmented_data, augmented_labels


    def reshape_time_series_data_v8(input_data, label_data, segments_per_new_sample, segment_length):
        """
        Reshape time series data and corresponding labels into a specified shape.

        :param input_data: Original time series data array.
        :param label_data: Corresponding labels for the data.
        :param segments_per_new_sample: Number of segments per new sample.
        :param segment_length: Length of each segment.

        :return: Reshaped data array and corresponding labels.
        """
        num_samples_original, num_channels, length_original = input_data.shape

        # Validate the feasibility of reshaping
        if length_original % segment_length != 0:
            raise ValueError("Segment length must evenly divide the original length.")

        total_segments_per_original_sample = (length_original // segment_length) * num_channels
        num_samples_new = (num_samples_original * total_segments_per_original_sample) // segments_per_new_sample

        # Validate if reshaping is possible
        if (num_samples_original * total_segments_per_original_sample) % segments_per_new_sample != 0:
            raise ValueError("Reshaping not possible with the given dimensions.")

        # Initialize reshaped data and labels
        new_shape = (num_samples_new, segments_per_new_sample, segment_length)
        reshaped_data = np.zeros(new_shape)
        reshaped_labels = np.zeros(num_samples_new)

        # Reshape the data and labels
        count = 0
        for i in range(num_samples_original):
            segment_count = 0
            for j in range(num_channels):
                for k in range(length_original // segment_length):
                    start_idx = k * segment_length
                    end_idx = start_idx + segment_length
                    reshaped_data[count, segment_count % segments_per_new_sample, :] = input_data[i, j, start_idx:end_idx]
                    if (segment_count + 1) % segments_per_new_sample == 0:
                        reshaped_labels[count] = label_data[i]  # Assign corresponding label
                        count += 1
                    segment_count += 1

        return reshaped_data, reshaped_labels
    
if __name__ == '__main__':
    path = "./Chuong Duong"
    input_data, output_labels = Data_process.data_loader(path)
    augmented_data, augmented_labels = Data_process.augment_time_series_data(input_data, output_labels)
    segments_per_new_sample = 8
    segment_length = 5000
    reshaped_data, reshaped_labels = Data_process.reshape_time_series_data_v8(augmented_data, augmented_labels, segments_per_new_sample, segment_length)


    
    # Select the data at index (1, 1, :) which has a shape of (8000,)
    Data = reshaped_data[200, :, :] #150 -> 200
    print(Data.shape)
    # Create the plot
    fig, axes = plt.subplots(reshaped_data.shape[1], 1, figsize=(15, 8), sharex=True)

    title_font = {'family': 'Times New Roman', 'size': 16, 'weight': 'bold'}
    label_font = {'family': 'Times New Roman', 'size': 14}
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['font.family'] = 'sans-serif'

    # Plot the data for each sub-array
    for i, ax in enumerate(axes):
        ax.plot(Data[i, :], linewidth=1, color = 'r')
        # ax.set_title(f'Z24 Signal Data at Index (1, {i}, :)', fontsize=12)
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.grid(True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        ax.set_xlim(-100, Data.shape[1]+100)
    # Set common labels using axes
    axes[-1].set_xlabel('Điểm dữ liệu', fontsize=14, fontdict=label_font)
    axes[0].set_title('Dữ liệu cầu Chương Dương chia nhỏ', fontsize=16, fontdict=title_font)

    # Create a "super" axis for the common Y-label and make it invisible
    super_ax = fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    super_ax.set_ylabel("Biên độ", fontsize=14, labelpad=15, fontdict=label_font)

    # Move the super axis ylabel to avoid overlap with subplots
    super_ax.yaxis.set_label_coords(-0.06,0.5)

    # Adjust the layout so that plots do not overlap
    plt.tight_layout()
    plt.show()