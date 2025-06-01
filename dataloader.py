import os
from scipy.io import loadmat
from scipy.interpolate import interp1d
directory = 'Dataset/Chuong Duong'

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
        print(filepath)
        print(mat_data)