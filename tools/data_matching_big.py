import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
import torchvision
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np

def prepare_new_dataloaders(dataloader1, dataloader2, batch_size, num_classes):
    # 统计dataloader2中每个类别的数量
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    for _, labels in dataloader2:
        for label in labels:
            class_counts[label] += 1

    # 从dataloader1中挑选每个类别的2张图片
    selected_images = {k: [] for k in range(num_classes)}
    counts = {k: 0 for k in range(num_classes)}

    for images, labels in dataloader1:
        for image, label in zip(images, labels):
            label = label.item()  # 转换为Python整数
            if counts[label] < 1:
                selected_images[label].append(image)
                counts[label] += 1
            if all(count >= 1 for count in counts.values()):
                break
        if all(count >= 1 for count in counts.values()):
            break

    # 创建一个字典，存储每个类别对应的其他类别的图片索引
    dataloader2_dataset = dataloader2.dataset
    other_images_indices = {label: [] for label in range(num_classes)}
    for idx in range(len(dataloader2_dataset)):
        _, lbl = dataloader2_dataset[idx]
        lbl = lbl.item()
        for label in range(num_classes):
            if lbl != label:
                other_images_indices[label].append(idx)

    # 计算每个类别对应的其他类别图片数量
    num_for_label_cal = torch.zeros(num_classes, dtype=torch.int64)
    for label in range(num_classes):
        num_for_label_cal[label] = len(other_images_indices[label]) * len(selected_images[label])

    # 创建自定义Dataset
    class CustomDataset(Dataset):
        def __init__(self, selected_images, other_images_indices, dataloader2_dataset, num_classes):
            self.selected_images = selected_images
            self.other_images_indices = other_images_indices
            self.dataloader2_dataset = dataloader2_dataset
            self.num_classes = num_classes

            # 构建索引列表
            self.indices = []
            for label in range(num_classes):
                for image_idx in range(len(self.selected_images[label])):
                    for other_idx in self.other_images_indices[label]:
                        self.indices.append((label, image_idx, other_idx))

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            label, image_idx, other_idx = self.indices[idx]
            # 获取selected_image和label
            selected_image = self.selected_images[label][image_idx]
            selected_label = label

            # 获取other_image和label
            other_image, other_label = self.dataloader2_dataset[other_idx]

            return (selected_image, selected_label), (other_image, other_label)

    custom_dataset = CustomDataset(selected_images, other_images_indices, dataloader2_dataset, num_classes)

    # 自定义collate_fn，将batch拆分为两个DataLoader
    def custom_collate_fn(batch):
        images1 = torch.stack([item[0][0] for item in batch])
        labels1 = torch.tensor([item[0][1] for item in batch], dtype=torch.long)
        images2 = torch.stack([item[1][0] for item in batch])
        labels2 = torch.tensor([item[1][1] for item in batch], dtype=torch.long)
        return images1, labels1, images2, labels2

    new_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 为了与原函数的输出一致，拆分new_dataloader的输出
    class SplitDataLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader

        def __iter__(self):
            for images1, labels1, images2, labels2 in self.dataloader:
                yield (images1, labels1), (images2, labels2)

        def __len__(self):
            return len(self.dataloader)

    new_dataloader1 = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: ((torch.stack([item[0][0] for item in x]), torch.tensor([item[0][1] for item in x], dtype=torch.long))))
    new_dataloader2 = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: ((torch.stack([item[1][0] for item in x]), torch.tensor([item[1][1] for item in x], dtype=torch.long))))

    return new_dataloader1, new_dataloader2, num_for_label_cal

'''
def prepare_new_dataloaders(dataloader1, dataloader2, batch_size, num_classes):
    # 统计dataloader2中每个类别的数量
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    for _, labels in dataloader2:
        for label in labels:
            class_counts[label] += 1


    # 从dataloader1中挑选每个类别的10张图片
    selected_images = {k: [] for k in range(num_classes)}
    counts = {k: 0 for k in range(num_classes)}

    for images, labels in dataloader1:
        for image, label in zip(images, labels):
            label = label.item()  # 转换为Python整数
            if counts[label] < 2:
                selected_images[label].append(image)
                counts[label] += 1
            if all(count >= 2 for count in counts.values()):
                break

    # 创建新的dataloaders的数据集
    new_images1 = []
    new_labels1 = []
    new_images2 = []
    new_labels2 = []

    num_for_label_cal = torch.zeros(num_classes, dtype=torch.int64)
    for label in range(num_classes):
        other_images = []
        other_labels = []

        # 收集除当前类别外的其他所有类别的图片和标签
        for images, labels in dataloader2:
            for image, lbl in zip(images, labels):
                if lbl != label:
                    other_images.append(image)
                    other_labels.append(lbl)
        num_for_label_cal[label] = len(other_images)
        # 对每个选中的图片进行复制
        for image in selected_images[label]:
            new_images1.extend([image] * len(other_images))
            new_labels1.extend([label] * len(other_images))

            # 复制dataloader2中的图片，每张图片重复10次
            new_images2.extend(other_images)
            new_labels2.extend(other_labels)

    # 转换为TensorDataset
    new_dataset1 = TensorDataset(torch.stack(new_images1), torch.tensor(new_labels1))
    new_dataset2 = TensorDataset(torch.stack(new_images2), torch.tensor(new_labels2))

    # 创建新的DataLoaders
    new_dataloader1 = DataLoader(new_dataset1, batch_size=batch_size, shuffle=False)
    new_dataloader2 = DataLoader(new_dataset2, batch_size=batch_size, shuffle=False)

    return new_dataloader1, new_dataloader2, num_for_label_cal


'''

