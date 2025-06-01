import os
import numpy as np
import random
from scipy.io import loadmat
from torch.utils.data import Dataset

class CrossFormerTimeSeriesDataset(Dataset):
    def __init__(self, mat_dir, keys_to_stack, seg_len=1024, segments_per_sample=1,
                 augment=True, img_size=(32, 32)):
        super().__init__()

        self.img_size = img_size

        # Load all .mat files and extract acceleration
        all_data = {}
        for filename in os.listdir(mat_dir):
            if filename.endswith('.mat'):
                path = os.path.join(mat_dir, filename)
                mat = loadmat(path)
                key = os.path.splitext(filename)[0]
                all_data[key] = mat['acceleration']

        # Stack selected keys
        input_data = np.stack([all_data[key] for key in keys_to_stack], axis=0)
        labels = np.arange(len(keys_to_stack))

        # Truncate to consistent length if needed
        input_data = input_data[:, :, :40000]  # (samples, channels, time)

        # Augmentation
        if augment:
            input_data, labels = self.augment_time_series_data(input_data, labels)

        # Reshape (segment)
        reshaped_data, reshaped_labels = self.reshape_time_series(input_data, labels,
                                                                  segments_per_sample, seg_len)

        # Convert 1D segments to 2D pseudo-images
        self.images = self.reshape_to_images(reshaped_data)
        self.labels = reshaped_labels.astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def augment_time_series_data(self, data, labels, num_augment=5):
        augmented = []
        augmented_labels = []
        for i in range(len(data)):
            for _ in range(num_augment):
                aug_type = random.choice(['noise', 'reverse', 'crop_pad'])
                x = data[i].copy()
                if aug_type == 'noise':
                    noise = np.random.normal(0, 0.001, x.shape)
                    x += noise
                elif aug_type == 'reverse':
                    x = np.flip(x, axis=-1)
                elif aug_type == 'crop_pad':
                    crop = random.randint(0, x.shape[-1] // 100)
                    x = np.pad(x, ((0, 0), (crop, 0)), mode='constant')[:, :-crop]
                augmented.append(x)
                augmented_labels.append(labels[i])
        return np.array(augmented), np.array(augmented_labels)

    def reshape_time_series(self, data, labels, segs_per_sample, seg_len):
        num_samples, num_channels, total_len = data.shape
        total_segs = (total_len // seg_len) * num_channels
        new_samples = (num_samples * total_segs) // segs_per_sample

        X = np.zeros((new_samples, segs_per_sample, seg_len))
        Y = np.zeros(new_samples)
        count = 0

        for i in range(num_samples):
            seg_count = 0
            for c in range(num_channels):
                for k in range(total_len // seg_len):
                    start = k * seg_len
                    end = start + seg_len
                    X[count, seg_count % segs_per_sample] = data[i, c, start:end]
                    if (seg_count + 1) % segs_per_sample == 0:
                        Y[count] = labels[i]
                        count += 1
                    seg_count += 1
        return X, Y

    def reshape_to_images(self, segments):
        H, W = self.img_size
        img_data = []
        for seg in segments:
            flat = seg.flatten()[:H * W]
            img = flat.reshape(H, W)
            img_data.append(img[np.newaxis, :, :])  # Add channel dim
        return np.array(img_data, dtype=np.float32)
