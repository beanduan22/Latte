import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
import random


def create_merged_dataloader(data_loader1, data_loader2, label_num, n_per_label):
    # 从 DataLoader 中获取 Dataset
    dataset1 = data_loader1.dataset
    dataset2 = data_loader2.dataset

    combined_dataset = ConcatDataset([dataset1, dataset2])

    label_to_data = {i: [] for i in range(label_num)}
    for data, target in combined_dataset:
        label_to_data[target.item()].append((data, target))

    final_data = []

    for label, items in label_to_data.items():
        if len(items) > n_per_label:
            selected_items = random.sample(items, n_per_label)
        else:
            selected_items = items

        final_data.extend(selected_items)

    class SortedDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

    sorted_dataset = SortedDataset(final_data)

    merged_dataloader = DataLoader(sorted_dataset, batch_size=64, shuffle=False)
    return merged_dataloader

def select_n_samples_per_class(error_loader, n, num_classes):
    # 初始化一个字典来存储每个标签的数据
    selected_data = {i: [] for i in range(num_classes)}
    selected_labels = {i: [] for i in range(num_classes)}

    # 遍历 error_loader 并根据标签选择 n 个样本
    for data, labels in error_loader:
        for d, label in zip(data, labels):
            label = label.item()  # 将标签转换为整数
            if len(selected_data[label]) < n:  # 如果还没有达到 n 个样本
                selected_data[label].append(d)
                selected_labels[label].append(label)
            if all(len(selected_data[i]) == n for i in range(num_classes)):
                break  # 如果所有标签都已经收集到 n 个样本，则停止

    # 将数据和标签从字典转化为张量
    data_list = []
    label_list = []
    for i in range(num_classes):
        data_list.extend(selected_data[i])
        label_list.extend(selected_labels[i])

    # 创建新的 TensorDataset
    new_data = torch.stack(data_list)
    new_labels = torch.tensor(label_list)

    new_dataset = TensorDataset(new_data, new_labels)
    new_loader = DataLoader(new_dataset, batch_size=error_loader.batch_size, shuffle=True)

    return new_loader

