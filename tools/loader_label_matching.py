import torch
from torch.utils.data import DataLoader, TensorDataset


class DataLoaderProcessor:
    def __init__(self, correct_loader, error_loader, num_labels):
        self.correct_loader = correct_loader
        self.error_loader = error_loader
        self.num_labels = num_labels

    def process_loaders(self):
        # Gather and sort the correct_loader by labels
        correct_data, correct_labels = self._gather_and_sort_data(self.correct_loader)
        # Repeat data
        repeated_correct_data = correct_data.repeat_interleave(9, dim=0)
        repeated_correct_labels = correct_labels.repeat_interleave(9, dim=0)

        # Gather and sort the error_loader by labels
        error_data, error_labels = self._gather_and_sort_data(self.error_loader)

        # Create matched error data
        matched_error_data = []
        matched_error_labels = []

        error_index = 0
        for label in repeated_correct_labels:
            while True:
                error_label = error_labels[error_index % len(error_labels)]
                if error_label != label:
                    matched_error_data.append(error_data[error_index % len(error_data)])
                    matched_error_labels.append(error_label)
                    error_index += 1
                    break
                error_index += 1

        # Create DataLoader from matched data
        matched_error_dataset = TensorDataset(torch.stack(matched_error_data), torch.tensor(matched_error_labels))
        matched_error_loader = DataLoader(matched_error_dataset, batch_size=64, shuffle=False)

        correct_dataset = TensorDataset(repeated_correct_data, repeated_correct_labels)
        correct_loader = DataLoader(correct_dataset, batch_size=64, shuffle=False)

        return correct_loader, matched_error_loader

    def _gather_and_sort_data(self, loader):
        all_data = []
        all_labels = []
        for data, labels in loader:
            all_data.append(data)
            all_labels.append(labels)

        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Sort by labels
        sorted_indices = torch.argsort(all_labels)
        sorted_data = all_data[sorted_indices]
        sorted_labels = all_labels[sorted_indices]

        return sorted_data, sorted_labels


# 示例用法
# 假设correct_loader和error_loader已定义

