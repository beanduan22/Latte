import torch
from torch.utils.data import DataLoader, Dataset

class DatasetSaverLoader:
    def __init__(self,  save_path):
        #self.dataset1 = dataset1
        #self.dataset2 = dataset2
        #self.filepath1 = save_path + '/error_data_base.pth'
        #self.filepath2 = save_path + '/correct_data_base.pth'
        #self.filepath1 = save_path + '/correct_data.pth'
        #self.filepath2 = save_path + '/error_data.pth'

        #self.filepath1 = save_path + '/error_base_data.pth'
        #self.filepath2 = save_path + '/error_inter_data.pth'
        self.filepath1 = save_path + '/error_big_data20.pth'
        self.filepath2 = save_path + '/error_inter_big_data20.pth'

    def save_datasets(self, dataset1, dataset2,):
        """ 保存两个数据集到指定的文件路径 """
        torch.save(dataset1, self.filepath1)
        torch.save(dataset2, self.filepath2)
        print(f"Datasets saved to {self.filepath1} and {self.filepath2}")

    def load_datasets(self):
        """ 从文件中加载两个数据集 """
        loaded_dataset1 = torch.load(self.filepath1)
        loaded_dataset2 = torch.load(self.filepath2)
        print(f"Datasets loaded from {self.filepath1} and {self.filepath2}")
        return loaded_dataset1, loaded_dataset2




import torch


def save_dataset_loader(loader, save_path):
    # 获取 DataLoader 中的 dataset（TensorDataset），并提取数据和标签
    data, labels = [], []
    save_path = save_path + '/date_loader.pth'
    for batch_data, batch_labels in loader:
        data.append(batch_data)
        labels.append(batch_labels)

    # 将所有批次的数据拼接起来
    data = torch.cat(data)
    labels = torch.cat(labels)

    # 将数据和标签保存为字典
    torch.save({'data': data, 'labels': labels}, save_path)
    print(f"Loader saved at {save_path}")


from torch.utils.data import DataLoader, TensorDataset


def load_dataset_loader(save_path, batch_size, shuffle=True):
    # 加载保存的字典
    save_path = save_path + '/date_loader.pth'
    saved_data = torch.load(save_path)

    # 获取数据和标签
    data = saved_data['data']
    labels = saved_data['labels']

    # 创建 TensorDataset
    dataset = TensorDataset(data, labels)

    # 创建新的 DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"Loader loaded from {save_path}")
    return loader

def save_dataset1_loader(loader, save_path):
    # 获取 DataLoader 中的 dataset（TensorDataset），并提取数据和标签
    data, labels = [], []
    save_path = save_path + '/date_loader1.pth'
    for batch_data, batch_labels in loader:
        data.append(batch_data)
        labels.append(batch_labels)

    # 将所有批次的数据拼接起来
    data = torch.cat(data)
    labels = torch.cat(labels)

    # 将数据和标签保存为字典
    torch.save({'data': data, 'labels': labels}, save_path)
    print(f"Loader saved at {save_path}")


from torch.utils.data import DataLoader, TensorDataset


def load_dataset1_loader(save_path, batch_size, shuffle=True):
    # 加载保存的字典
    save_path = save_path + '/date_loader1.pth'
    saved_data = torch.load(save_path)

    # 获取数据和标签
    data = saved_data['data']
    labels = saved_data['labels']

    # 创建 TensorDataset
    dataset = TensorDataset(data, labels)

    # 创建新的 DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    print(f"Loader loaded from {save_path}")
    return loader

# 示例使用方式

