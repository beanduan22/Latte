import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import sqrtm
from PIL import Image


def calculate_fid(images1, images2):
    #def prepare_images(images):
    #    processed_images = []
    #    for img in images:
    #        img = TF.resize(img, (299, 299))  # 调整到Inception的输入尺寸
    #        #img = TF.to_tensor(img)
    #        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #        processed_images.append(img)
    #    return torch.stack(processed_images)

    def get_features(model, dataloader):
        model.eval()
        features = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, list) or isinstance(batch, tuple):
                    batch = batch[0]
                batch = batch.to(device)
                batch_features = model(batch)
                features.append(batch_features)
        return torch.cat(features, dim=0).cpu().numpy()

    # 处理图像
    #images1 = prepare_images(list_of_images1)
    #images2 = prepare_images(list_of_images2)
    batch_size = 32
    dataset1 = TensorDataset(images1)
    dataset2 = TensorDataset(images2)
    dataloader1 = DataLoader(dataset1, batch_size=batch_size)
    dataloader2 = DataLoader(dataset2, batch_size=batch_size)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()

    # 获取特征
    features1 = get_features(model, dataloader1)
    features2 = get_features(model, dataloader2)

    # 计算统计量
    mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)

    # 计算 Fréchet 距离
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2), disp=False)[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid

# 用法示例
# list_of_images1 = [Image.open('path_to_image1.jpg'), Image.open('path_to_image2.jpg'), ...]
# list_of_images2 = [Image.open('path_to_image3.jpg'), Image.open('path_to_image4.jpg'), ...]
# fid_score = calculate_fid(list_of_images1, list_of_images2)
# print("FID score:", fid_score)
