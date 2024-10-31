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
from collections import Counter

class PredictedStatistics:
    def __init__(self, thresholds):
        self.thresholds = thresholds  # 阈值列表
        #self.results_matrix = np.zeros((len(thresholds), 10))  # 存储结果的矩阵
        self.results_matrix = np.zeros((1, 10))  # 存储结果的矩阵
        self.current_count = 0
        self.current_set = set()
        self.threshold_index = 0
        self.repeat_index = 0
        self.total_collected = 0  # 总共收集的predicted数量

    def update(self, predicted):
        """ 更新统计数据 """
        current_batch_count = len(predicted)
        self.current_count += current_batch_count
        self.current_set.update(predicted)
        self.total_collected += current_batch_count

        # 检查是否达到当前阈值
        if self.current_count >= self.thresholds[self.threshold_index]:
            # 记录统计结果
            unique_count = len(self.current_set)
            self.results_matrix[self.threshold_index][self.repeat_index] = unique_count

            # 重置统计和索引
            self.current_count -= self.thresholds[self.threshold_index]
            self.current_set.clear()
            self.repeat_index += 1

            # 检查是否需要移动到下一个threshold
            if self.repeat_index == 10:
                self.repeat_index = 0
                self.threshold_index += 1
                if self.threshold_index >= len(self.thresholds):
                    return True  # 表示所有数据已处理完毕

        return False

    def get_results(self):
        """ 获取当前统计结果 """
        return self.results_matrix
def filter_tensor(tensor1, tensor2):
    # 找到所有相同位置值相同的位置
    mask = tensor1 != tensor2

    # 使用mask过滤第二个tensor，只保留值不同的位置
    filtered_tensor2 = tensor2[mask]

    return filtered_tensor2
# 全局字典，用于存储每个标签的累计出现次数
label_counts = Counter()

def count_labels(labels):
    # 将标签列表转换为扁平形式，因为输入假定为(batch, 1)
    flattened_labels = [label.item() if torch.is_tensor(label) and label.dim() == 0 else label for label in labels]

    # 计算当前传入的labels的出现次数
    current_counts = Counter(flattened_labels)

    # 更新全局字典
    global label_counts
    label_counts.update(current_counts)

    return dict(label_counts)


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

class ModelEvaluatorAfterTrain:
    def __init__(self, model, vqvae, device, logger, pic_dir):
        self.model = model.to(device)
        self.vqvae = vqvae.to(device)
        self.device = device
        self.logger = logger
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
                reconstructed_images, enc_outputs, vq_loss, perplexity, _ = self.vqvae(images)
                classification_outputs = self.model(images)
                enc_loss = self.criterion(classification_outputs, labels)
                dec_loss = self.mse_criterion(reconstructed_images, images)

                # 计算 FID
                #batch_fid = calculate_fid(prepare_images(images.cpu()), prepare_images(reconstructed_images.cpu()))
                #total_fid += batch_fid
                batch_ssim = calculate_ssim_4D(images.cpu(), reconstructed_images.cpu())
                total_ssim += batch_ssim

                total_enc_loss += enc_loss.item()
                total_dec_loss += dec_loss.item()
                total_vq_loss += vq_loss.item()  # Collecting VQ loss
                _, predicted = torch.max(classification_outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = total_correct / total
        error_rate = (1 - accuracy) * 100
        error_num = error_rate * total / 100
        avg_fid = total_fid / len(test_loader)
        avg_ssim = total_ssim / len(test_loader)

        self.logger.info(
            f'Test Loss - Encoder: {total_enc_loss / len(test_loader):.4f}, Decoder: {total_dec_loss / len(test_loader):.4f}, VQ Loss: {total_vq_loss / len(test_loader):.4f}, '
            f'Error_rate: {error_rate:.2f}%, Error_num: {error_num:.1f}, Average FID: {avg_fid:.2f}, Average SSIM: {avg_ssim:.2f}'
        )

    def evaluate_compare(self, x_adv_loader, error_loader, for_label_num):
        self.model.eval()
        self.vqvae.eval()
        total_dec_loss, total_correct, total, total_ssim = 0, 0, 0, 0
        stats = PredictedStatistics(for_label_num)
        # Assume that x_adv_loader and error_loader have the same length and batch sizes
        zipped_loaders = zip(x_adv_loader, error_loader)
        with torch.no_grad():
            for idx, ((x_latent, x_labels), (e_latent, e_labels)) in enumerate(zipped_loaders):
                x_latent, x_labels = x_latent.to(self.device), x_labels.to(self.device)
                e_latent, e_labels = e_latent.to(self.device), e_labels.to(self.device)
                x_z_q,_,_,_ = self.vqvae.quantize(x_latent)
                #x_z_q = self.vqvae.quant_to_dec(x_z_q)
                x_images = self.vqvae.decoder(x_z_q)
                e_latent = e_latent.squeeze(1)
                e_z_q,_,_,_ = self.vqvae.quantize(e_latent)
                #e_z_q = self.vqvae.quant_to_dec(e_z_q)
                e_images = self.vqvae.decoder(e_z_q)
                # Check if labels match
                if torch.equal(x_labels, e_labels):
                    # Calculate SSIM
                    ssim_value = calculate_ssim_4D(x_images.cpu(), e_images.cpu())
                    total_ssim += ssim_value

                    # Processing with encoder and decoder
                    classification_outputs = self.model(x_images)
                    #classification_outputs = self.encoder.fc(torch.flatten(enc_outputs, 1))
                    dec_loss = self.mse_criterion(x_images, e_images)

                    #total_enc_loss += enc_loss.item()
                    total_dec_loss += dec_loss.item()
                    _, predicted = torch.max(classification_outputs.data, 1)

                    total_correct += (predicted == x_labels).sum().item()
                    total += x_labels.size(0)
                    new_predicted = filter_tensor(x_labels, predicted)
                    count_labels(new_predicted)

                    # Visualization and logging
                    self.save_image_pair(x_images, e_images, x_labels, predicted, idx)
                    correct = (predicted == x_labels)
                    self.logger.info(
                        f'Batch {idx}: SSIM: {ssim_value:.4f}, Correct: {correct.tolist()}, '
                        f'Predicted: {predicted.tolist()}, Actual: {x_labels.tolist()}'
                    )

        accuracy = total_correct / total
        mse = total_dec_loss / total
        error_rate = (1 - accuracy) * 100
        error_num = error_rate * total / 100
        results_matrix = stats.get_results()
        ssim = total_ssim / len(x_adv_loader)

        # Logging the results
        self.logger.info(
            f' Test Loss - MSE: {mse:.4f}, '
            f'Accuracy: {accuracy:.2f}, Error Rate: {error_rate:.2f}%, Error Num: {error_num:.1f}'
            f'Labels: {label_counts}, '
            f'Labels_num_matrix_per_image: {results_matrix}, '
        )
        self.logger.info(
            f'Final SSIM: {ssim:.4f}, '
        )

    def save_image_pair(self, images1, images2, images1_label, images2_label, batch_idx):
        """ Saves a pair of images side by side for comparison """

        # Normalize images from potential [-1, 1] to [0, 1] range

        for i, (img1, img2, label1, label2) in enumerate(zip(images1, images2, images1_label, images2_label)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # Ensure images are clipped within [0, 1] range
            img1 = img1.cpu().squeeze().permute(1, 2, 0).numpy()
            img2 = img2.cpu().squeeze().permute(1, 2, 0).numpy()
            img1 = (img1 * 0.5) + 0.5
            img2 = (img2 * 0.5) + 0.5

            axes[0].imshow(img1, cmap='gray' if img1.shape[2] == 1 else None)
            axes[0].set_title('Adv/' + str(label2) )
            axes[1].imshow(img2, cmap='gray' if img2.shape[2] == 1 else None)
            axes[1].set_title('Original/' + str(label1))
            axes[0].axis('off')
            axes[1].axis('off')

            plt.savefig(f'{self.output_dir}/comparison_{batch_idx}_{i}.png')
            plt.close()
    def latent_compare(self, loader1, loader2):
        self.vqvae.eval()
        cosine_sim_list = []
        euclidean_dist_list = []

        with torch.no_grad():
            for (images1, labels1), (images2, labels2) in zip(loader1, loader2):
                images1, images2 = images1.to(self.device), images2.to(self.device)
                if images1.size(0) > images2.size(0):
                    images1 = images1[:images2.size(0), :]
                elif images1.size(0) < images2.size(0):
                    images2 = images2[:images1.size(0), :]

                z_q_1, vq_loss1, p1, _ = self.vqvae.quantize(images1)
                decoded_images1 = self.vqvae.decoder(z_q_1)
                images2 = images2.squeeze(1)
                z_q_2, vq_loss2, p2, _ = self.vqvae.quantize(images2)
                decoded_images2 = self.vqvae.decoder(z_q_2)


                cosine_sim = calculate_cosine_similarity(decoded_images1, decoded_images2)
                euclidean_dist = calculate_euclidean_distance(decoded_images1, decoded_images2)

                cosine_sim_list.append(cosine_sim)
                euclidean_dist_list.append(euclidean_dist)

        # Log average similarity and distance
        avg_cosine_sim = sum(cosine_sim_list) / len(cosine_sim_list)
        avg_euclidean_dist = sum(euclidean_dist_list) / len(euclidean_dist_list)


