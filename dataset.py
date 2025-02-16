import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import torch
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, dataset):
        super(AudioDataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file_path = self.dataset.iloc[idx]['file_path']
        label = self.dataset.iloc[idx]['class']
        waveform, sample_rate = torchaudio.load(file_path)
        label = torch.tensor(label, dtype=torch.long)
        return waveform, label


def generate_dataset():
    dataset_path = 'clean_dataset/'

    dataset = pd.DataFrame(columns=['file_path', 'class'])
    classes = os.listdir(dataset_path)
    for cls in classes:
        files = os.listdir(os.path.join(dataset_path, cls))
        for file in files:
            dataset.loc[len(dataset)] = [os.path.join(dataset_path, cls, file), cls]

    label_encoder = LabelEncoder()
    dataset['class'] = label_encoder.fit_transform(dataset['class'])

    mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
    print("Mapping:", mapping)

    print(f'len dataset: {len(dataset)} from {len(classes)} classes')

    train_dataset, test_dataset = train_test_split(dataset, test_size=.2, random_state=42)
    test_dataset, validation_dataset = train_test_split(test_dataset, test_size=.25, random_state=42)

    train_dataset = AudioDataset(train_dataset)
    test_dataset = AudioDataset(test_dataset)
    validation_dataset = AudioDataset(validation_dataset)

    print(f'Train: {len(train_dataset)}')
    print(f'Validation: {len(validation_dataset)}')
    print(f'Test: {len(test_dataset)}')

    train_dataset = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_dataset = DataLoader(validation_dataset, batch_size=16, shuffle=False)
    test_dataset = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dataset, validation_dataset, test_dataset
