import torch
from torch.utils.data import TensorDataset, DataLoader

class DatasetManager:
    def __init__(self, dataset_path):
        """
        Initialize the DatasetManager with a path to save or load the dataset.
        :param dataset_path: Path to save or load the dataset.
        """
        self.dataset_path = dataset_path

    def save_dataset(self, dataset):
        """
        Saves the provided dataset to the specified path.
        :param dataset: A TensorDataset to be saved.
        """
        torch.save(dataset, self.dataset_path)
        print(f"Dataset saved to {self.dataset_path}")

    def load_dataset(self, batch_size=64, shuffle=False):
        """
        Loads a dataset from the specified path and returns a DataLoader.
        :param batch_size: Batch size for the DataLoader.
        :param shuffle: Whether to shuffle the data in the DataLoader.
        :return: DataLoader with the loaded dataset.
        """
        dataset = torch.load(self.dataset_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        print(f"Dataset loaded from {self.dataset_path}")
        return dataloader

# Usage
# Create a dataset

