import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import functional as TF
from tools.FID import calculate_fid
from tools.SSIM import calculate_ssim_4D
from tools.CCS import *
from tools.CED import *
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

def prepare_images(images):
    processed_images = []
    for img in images:
        # 检查图像是否已经是张量
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)

        # 确保图像尺寸正确
        if img.shape[1] != 299 or img.shape[2] != 299:  # (C, H, W)
            img = TF.resize(img, (299, 299))

        # 归一化
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        processed_images.append(img)

    # 堆叠成一个批处理张量
    return torch.stack(processed_images)

class ModelEvaluator:
    def __init__(self, model, vqvae, device, log, pic_dir):
        self.model = model.to(device)
        self.vqvae = vqvae.to(device)
        self.device = device
        self.logger = log
        self.criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        self.output_dir = pic_dir

    def evaluate(self, test_loader):
        self.model.eval()
        self.vqvae.eval()
        total_enc_loss, total_dec_loss, total_vq_loss, total_correct, total, total_fid, total_ssim = 0, 0, 0, 0, 0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                classification_outputs = self.model(images)

                cls_loss = self.criterion(classification_outputs, labels)

                reconstructed_images, encoder_out, vq_loss, perplexity, _ = self.vqvae(images)

                dec_loss = self.mse_criterion(reconstructed_images, images)

                # 计算 FID
                #batch_fid = calculate_fid(prepare_images(images.cpu()), prepare_images(reconstructed_images.cpu()))
                #total_fid += batch_fid
                batch_ssim = calculate_ssim_4D(images.cpu(), reconstructed_images.cpu())
                total_ssim += batch_ssim

                total_enc_loss += cls_loss.item()
                total_dec_loss += dec_loss.item()
                total_vq_loss += vq_loss.item()  # Collecting VQ loss

                _, predicted = torch.max(classification_outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = total_correct / total
        error_rate = (1 - accuracy) * 100
        error_num = error_rate * total / 100
        avg_ssim = total_ssim / len(test_loader)

        self.logger.info(
            f'Test Loss - Model CLS: {total_enc_loss / len(test_loader):.4f}, Decoder: {total_dec_loss / len(test_loader):.4f}, VQ Loss: {total_vq_loss / len(test_loader):.4f}, Perplexity: {perplexity:.4f}' 
            f'Error_rate: {error_rate:.2f}%, Error_num: {error_num:.1f}, Average SSIM: {avg_ssim:.2f}'
        )

    def evaluate_compare(self, x_adv_loader, error_loader):
        self.model.eval()
        self.vqvae.eval()
        total_enc_loss, total_dec_loss, total_vq_loss, total_correct, total, total_ssim = 0, 0, 0, 0, 0, 0

        # Assume that x_adv_loader and error_loader have the same length and batch sizes
        zipped_loaders = zip(x_adv_loader, error_loader)
        with torch.no_grad():
            for idx, ((x_images, x_labels), (e_images, e_labels)) in enumerate(zipped_loaders):
                x_images, x_labels = x_images.to(self.device), x_labels.to(self.device)
                e_images, e_labels = e_images.to(self.device), e_labels.to(self.device)

                # Check if labels match
                if torch.equal(x_labels, e_labels):
                    # Calculate SSIM
                    ssim_value = calculate_ssim_4D(x_images.cpu(), e_images.cpu())
                    total_ssim += ssim_value

                    # Processing with encoder and decoder
                    classification_outputs = self.model(x_images)
                    cls_loss = self.criterion(classification_outputs, x_labels)
                    reconstructed_images, encoder_out, vq_loss, perplexity, _ = self.vqvae(x_images)
                    dec_loss = self.mse_criterion(reconstructed_images, x_images)

                    total_enc_loss += cls_loss.item()
                    total_dec_loss += dec_loss.item()
                    total_vq_loss += vq_loss.item()  # Collecting VQ loss
                    _, predicted = torch.max(classification_outputs.data, 1)
                    total_correct += (predicted == x_labels).sum().item()
                    total += x_labels.size(0)

                    # Visualization and logging
                    self.save_image_pair(x_images, e_images, idx)
                    correct = (predicted == x_labels)
                    self.logger.info(
                        f'Batch {idx}: SSIM: {ssim_value:.4f}, Correct: {correct.tolist()}, '
                        f'Predicted: {predicted.tolist()}, Actual: {x_labels.tolist()}'
                    )

        accuracy = total_correct / total
        error_rate = (1 - accuracy) * 100
        error_num = error_rate * total / 100

        # Logging the results
        self.logger.info(
            f'Final SSIM: {total_ssim / len(x_adv_loader):.4f}, '
            f'Test Loss - Model CLS: {total_enc_loss / len(x_adv_loader):.4f}, '
            f'Decoder: {total_dec_loss / len(x_adv_loader):.4f}, '
            f'VQ Loss: {total_vq_loss / len(x_adv_loader):.4f}, '
            f'Perplexity: {perplexity:.4f}, '
            f'Accuracy: {accuracy:.2f}, Error Rate: {error_rate:.2f}%, Error Num: {error_num:.1f}'
        )

    def save_image_pair(self, images1, images2, batch_idx):
        """ Saves a pair of images side by side for comparison """

        # Normalize images from potential [-1, 1] to [0, 1] range

        for i, (img1, img2) in enumerate(zip(images1, images2)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # Ensure images are clipped within [0, 1] range
            img1 = img1.cpu().squeeze().permute(1, 2, 0).numpy()
            img2 = img2.cpu().squeeze().permute(1, 2, 0).numpy()
            img1 = (img1 * 0.5) + 0.5
            img2 = (img2 * 0.5) + 0.5

            axes[0].imshow(img1)
            axes[0].set_title('Original')
            axes[1].imshow(img2)
            axes[1].set_title('Reconstructed')
            axes[0].axis('off')
            axes[1].axis('off')

            plt.savefig(f'{self.output_dir}/comparison_{batch_idx}_{i}.png')
            plt.close()



    def get_input(self, test_loader):
        self.model.eval()
        self.vqvae.eval()
        correct_enc_outputs = []
        error_enc_outputs = []
        correct_labels = []
        error_labels = []
        total_ssim_correct, total_ssim_error = 0, 0
        num_correct_samples, num_error_samples = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                classification_outputs = self.model(images)
                reconstructed_images, enc_outputs, vq_loss, perplexity, _ = self.vqvae(images)
                _, predicted = torch.max(classification_outputs.data, 1)
                enc_outputs = enc_outputs.squeeze()
                # Determine correctness
                correct = predicted == labels
                incorrect = ~correct

                # Collect data for DataLoader
                correct_enc_outputs.append(enc_outputs[correct])
                error_enc_outputs.append(enc_outputs[incorrect])
                correct_labels.append(labels[correct])
                error_labels.append(labels[incorrect])

                # Collect images for SSIM calculation
                correct_images = images[correct]
                correct_reconstructions = reconstructed_images[correct]
                error_images = images[incorrect]
                error_reconstructions = reconstructed_images[incorrect]

                if len(correct_images) > 0:
                    ssim_correct = calculate_ssim_4D(correct_images.cpu(), correct_reconstructions.cpu())
                    total_ssim_correct += ssim_correct
                    num_correct_samples += len(correct_images)

                if len(error_images) > 0:
                    ssim_error = calculate_ssim_4D(error_images.cpu(), error_reconstructions.cpu())
                    total_ssim_error += ssim_error
                    num_error_samples += len(error_images)

        # Convert collected data into tensors
        correct_enc_outputs = torch.cat(correct_enc_outputs)
        error_enc_outputs = torch.cat(error_enc_outputs)
        correct_labels = torch.cat(correct_labels)
        error_labels = torch.cat(error_labels)

        # Calculate accuracies and averages
        accuracy = len(correct_labels) / (len(correct_labels) + len(error_labels))
        avg_ssim_correct = total_ssim_correct/ len(test_loader)
        avg_ssim_error = total_ssim_error/ len(test_loader)

        # Prepare DataLoader for correct and error classifications
        correct_dataset = TensorDataset(correct_enc_outputs, correct_labels)
        error_dataset = TensorDataset(error_enc_outputs, error_labels)
        correct_loader = DataLoader(correct_dataset, batch_size=16, shuffle=False)
        error_loader = DataLoader(error_dataset, batch_size=16, shuffle=False)

        # Log the information
        self.logger.info(
            f'Total correct inputs: {len(correct_labels)}, Total error inputs: {len(error_labels)}, '
            f'Accuracy: {accuracy * 100:.2f}%, '
            f'Average SSIM (Correct classifications): {avg_ssim_correct:.2f}, '
            f'Average SSIM (Error classifications): {avg_ssim_error:.2f}'
        )

        return correct_loader, error_loader

    def latent2images(self, test_loader):
        self.vqvae.eval()  # 确保解码器处于评估模式
        x_adv_list = []  # 用来存储解码后的图像
        labels_list = []  # 用来存储对应的标签

        with torch.no_grad():  # 在这个块内部的计算不会进行梯度追踪
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.unsqueeze(2).unsqueeze(3)
                x_adv = self.vqvae(images)

                x_adv_list.append(x_adv)  # 使用append方法向列表中添加元素
                labels_list.append(labels)

        # 将收集的列表转换成张量，并通过torch.cat连接成一个完整的张量
        x_adv_tensor = torch.cat(x_adv_list)
        labels_tensor = torch.cat(labels_list)

        # 使用TensorDataset和DataLoader创建新的loader
        new_dataset = TensorDataset(x_adv_tensor, labels_tensor)
        new_loader = DataLoader(new_dataset, batch_size=16, shuffle=False)

        return new_loader
    def latent_compare(self, loader1, loader2):
        self.vqvae.eval()  # 确保解码器处于评估模式
        cosine_sim_list = []  # 存储余弦相似度
        euclidean_dist_list = []  # 存储欧氏距离

        with torch.no_grad():  # 在这个块内部的计算不会进行梯度追踪
            for (images1, labels1), (images2, labels2) in zip(loader1, loader2):
                images1, images2 = images1.to(self.device), images2.to(self.device)

                # Decode images
                #images1 = images1.unsqueeze(2).unsqueeze(3)
                #images2 = images2.unsqueeze(2).unsqueeze(3)
                decoded_images1 = self.vqvae(images1)
                decoded_images2 = self.vqvae(images2)

                # Calculate cosine similarity and euclidean distance
                cosine_sim = calculate_cosine_similarity(decoded_images1, decoded_images2)
                euclidean_dist = calculate_euclidean_distance(decoded_images1, decoded_images2)

                cosine_sim_list.append(cosine_sim)
                euclidean_dist_list.append(euclidean_dist)

        # Log average similarity and distance
        avg_cosine_sim = sum(cosine_sim_list) / len(cosine_sim_list)
        avg_euclidean_dist = sum(euclidean_dist_list) / len(euclidean_dist_list)
        self.logger.info(
            f'Average Cosine Similarity: {avg_cosine_sim:.4f}, Average Euclidean Distance: {avg_euclidean_dist:.4f}')

