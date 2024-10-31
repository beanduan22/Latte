import torch
import matplotlib.pyplot as plt
import numpy as np


class ModelVisualizer:
    def __init__(self, encoder, decoder, device):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device
    def visualize_reconstruction(self, data_loader):
        """从 DataLoader 中随机抽取一张图片，显示原始和重建图像。"""
        self.encoder.eval()
        self.decoder.eval()

        images, _ = next(iter(data_loader))
        image = images[0:1].to(self.device)

        with torch.no_grad():
            encoded = self.encoder.features(image)
            reconstructed = self.decoder(encoded)

        original = image.cpu().squeeze().permute(1, 2, 0).numpy()
        original = (original * 0.5) + 0.5  # 反归一化
        reconstructed_image = reconstructed.cpu().squeeze().permute(1, 2, 0).numpy()
        reconstructed_image = (reconstructed_image * 0.5) + 0.5  # 反归一化

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray' if original.shape[2] == 1 else None)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray' if reconstructed_image.shape[2] == 1 else None)
        plt.title('Reconstructed Image')
        plt.axis('off')

        plt.show()


    def visualize_latent(self, data_loader1, data_loader2):
        """从 DataLoader 中随机抽取一张图片，显示原始和重建图像。"""
        self.encoder.eval()
        self.decoder.eval()

        images1, _ = next(iter(data_loader1))
        image1 = images1[0:1].to(self.device)

        images2, _ = next(iter(data_loader2))
        image2 = images2[0:1].to(self.device)

        with torch.no_grad():
            image1 = image1.unsqueeze(2).unsqueeze(3)
            reconstructed = self.decoder(image1)

            image2 = image2.unsqueeze(2).unsqueeze(3)
            orginal = self.decoder(image2)


        original = orginal.cpu().squeeze().permute(1, 2, 0).numpy()
        original = (original * 0.5) + 0.5  # 反归一化
        reconstructed_image = reconstructed.cpu().squeeze().permute(1, 2, 0).numpy()
        reconstructed_image = (reconstructed_image * 0.5) + 0.5  # 反归一化

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray' if original.shape[2] == 1 else None)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray' if reconstructed_image.shape[2] == 1 else None)
        plt.title('Reconstructed Image')
        plt.axis('off')

        plt.show()
