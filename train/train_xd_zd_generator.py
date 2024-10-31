import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import ConcatDataset
def expand_dataset(loader, expansion_factor):
    dataset_list = [loader.dataset for _ in range(expansion_factor)]
    combined_dataset = ConcatDataset(dataset_list)
    return DataLoader(combined_dataset, batch_size=loader.batch_size, shuffle=True)

class AdversarialTraining:
    def __init__(self, dnnmodel, vqvae, zd, xd, save_path, logger, enc_loss_weight, dec_loss_weight, vq_loss_weight,
                 lr_encoder, lr_zd, lr_xd,
                 weight_zd=0.0, weight_xd=0.0, weight_recon=0.0, device='cuda'):
        self.dnnmodel = dnnmodel.to(device)
        self.vqvae = vqvae.to(device)
        self.zd = zd.to(device)
        self.xd = xd.to(device)
        self.save_path = save_path
        self.logger = logger
        self.device = device
        self.weight_zd = weight_zd
        self.weight_xd = weight_xd
        self.weight_recon = weight_recon
        self.enc_loss_weight = enc_loss_weight
        self.dec_loss_weight = dec_loss_weight
        self.vq_loss_weight = vq_loss_weight
        self.criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()
        self.optimizer_model = torch.optim.Adam([{"params": self.dnnmodel.parameters(), "lr": lr_encoder}])

        self.optimizer_vqvae = torch.optim.Adam([{"params": self.vqvae.parameters(), "lr": lr_encoder}])

        # OptimizersZ
        #self.optimizer_en_de = torch.optim.Adam(encoder.parameters(), lr=lr_encoder)
        self.optimizer_zd = torch.optim.Adam(zd.parameters(), lr=lr_zd)
        self.optimizer_xd = torch.optim.Adam(xd.parameters(), lr=lr_xd)

    def save_xd_model(self):
        torch.save(self.xd.state_dict(), f'{self.save_path}/xd_model.pth')
        self.logger.info("XD model saved successfully.")

    def save_zd_model(self):
        torch.save(self.zd.state_dict(), f'{self.save_path}/zd_model.pth')
        self.logger.info("ZD model saved successfully.")

    def train(self, loader_1, loader_2, train_loader, test_loader, epochs, lambda_, epsilon=0.01,
                  delta=0.1, ):
        for epoch in range(epochs):
            self.train_encoders(train_loader, test_loader)
            z_adv_loader = self.train_discriminators_generators_other_label(loader_1, loader_2, lambda_)
            self.logger.info(f"Epoch {epoch + 1}/{epochs} completed")
            if epoch == epochs - 1 :
                return z_adv_loader

    def train_discriminators_generators_other_label(self, loader_1, loader_2, lambda_, method='interpolation'):
        bce_loss = nn.BCEWithLogitsLoss()
        z_adv_list = []
        labels_list = []
        z_adv_total = torch.Tensor().to(dtype=torch.float32)
        labels_total = torch.Tensor().to(dtype=torch.long)

        for (z_normal, label), (z_error, _) in zip(loader_1, loader_2):
            z_normal = z_normal.squeeze(1)
            z_error = z_error.squeeze(1)
            if len(z_error) != len(z_normal):
                z_error = z_error[:len(z_normal)]
                label = label[:len(z_error)]
            z_normal = z_normal.to(self.device)
            z_error = z_error.to(self.device)
            label = label.to(self.device)

            if method == 'interpolation':
                z_adv = self.generate_z_new_interpolation(z_normal, z_error, lambda_)

            z_adv_list.append(z_adv)
            labels_list.append(label)

            if len(z_adv_list) >= 1000:
                # 合并这批数据
                z_adv_batch = torch.cat(z_adv_list)
                labels_batch = torch.cat(labels_list)

                z_adv_total = torch.cat((z_adv_total, z_adv_batch)) if z_adv_total.nelement() > 0 else z_adv_batch
                labels_total = torch.cat((labels_total, labels_batch)) if labels_total.nelement() > 0 else labels_batch

                z_adv_list = []
                labels_list = []

            self.optimizer_zd.zero_grad()
            zd_real_loss = bce_loss(self.zd(z_normal), torch.ones(z_normal.size(0), 1, device=self.device))
            zd_fake_loss = bce_loss(self.zd(z_adv.detach()), torch.zeros(z_adv.size(0), 1, device=self.device))
            zd_loss = zd_real_loss + zd_fake_loss
            zd_loss.backward()
            self.optimizer_zd.step()


            self.optimizer_xd.zero_grad()
            recon_normal_q,_,_,_ = self.vqvae.quantize(z_normal)
            #recon_normal_q = self.vqvae.quant_to_dec(recon_normal_q)
            recon_normal = self.vqvae.decoder(recon_normal_q)

            recon_adv_q, _, _, _ = self.vqvae.quantize(z_adv.detach())
            #recon_adv_q = self.vqvae.quant_to_dec(recon_adv_q)
            recon_adv = self.vqvae.decoder(recon_adv_q)

            xd_real_loss = bce_loss(self.xd(recon_normal), torch.ones(z_normal.size(0), 1, device=self.device))
            xd_fake_loss = bce_loss(self.xd(recon_adv), torch.zeros(z_adv.size(0), 1, device=self.device))
            xd_loss = xd_real_loss + xd_fake_loss
            xd_loss.backward()
            self.optimizer_xd.step()

            # 更新生成器（解码器）
            self.optimizer_vqvae.zero_grad()

            recon_adv_q, _, _, _ = self.vqvae.quantize(z_adv)
            #recon_adv_q = self.vqvae.quant_to_dec(recon_adv_q)
            recon_adv = self.vqvae.decoder(recon_adv_q)

            adv_loss = -bce_loss(self.xd(recon_adv), torch.ones(z_normal.size(0), 1, device=self.device))
            adv_loss.backward()
            self.optimizer_vqvae.step()

        if z_adv_list:
            z_adv_batch = torch.cat(z_adv_list)
            z_adv_total = z_adv_total.to(z_adv_batch.device)
            labels_batch = torch.cat(labels_list)
            labels_total = labels_total.to(labels_batch.device)
            z_adv_total = torch.cat((z_adv_total, z_adv_batch))
            labels_total = torch.cat((labels_total, labels_batch))
        dataloader_zadv = DataLoader(TensorDataset(z_adv_total, labels_total), batch_size=16, shuffle=False)
        return dataloader_zadv

    def test(self, test_loader):
        self.dnnmodel.eval()
        self.vqvae.eval()

        total_enc_loss, total_dec_loss, total_vq_loss, total_correct, total = 0, 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                classification_outputs = self.dnnmodel(images)

                reconstructed_images, encoder_out, vq_loss, perplexity, _ = self.vqvae(images)

                enc_loss = self.criterion(classification_outputs, labels)
                dec_loss = self.mse_criterion(reconstructed_images, images)

                # Log losses and compute accuracy
                total_enc_loss += enc_loss.item()
                total_dec_loss += dec_loss.item()
                total_vq_loss += vq_loss.item()  # Collecting VQ loss
                _, predicted = torch.max(classification_outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * total_correct / total
        self.logger.info(
            f'Test Loss - Encoder: {total_enc_loss / len(test_loader):.4f}, Decoder: {total_dec_loss / len(test_loader):.4f}, VQ Loss: {total_vq_loss / len(test_loader):.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.2f}%')

    def train_encoders(self, train_loader, test_loader):
        self.logger.info("training VQ-VAE encoder-decoder")
        self.dnnmodel.train()
        self.vqvae.train()

        total_enc_loss, total_dec_loss, total_vq_loss, total_correct, total = 0, 0, 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer_vqvae.zero_grad()
            # Encoder outputs
            classification_outputs = self.dnnmodel(images)

            # Pass encoder outputs through the vector quantizer
            reconstructed_images, encoder_out, vq_loss, perplexity, _ = self.vqvae(images)

            # Calculate losses
            #classification_outputs = self.encoder.fc(torch.flatten(enc_outputs, 1))
            enc_loss = self.criterion(classification_outputs, labels)  # Classification loss
            dec_loss = self.mse_criterion(reconstructed_images, images)  # Reconstruction loss

            # Total loss includes VQ loss
            total_loss = self.dec_loss_weight * dec_loss + self.vq_loss_weight * vq_loss
            total_loss.backward()

            self.optimizer_vqvae.step()

            # Logging and metrics calculation
            total_enc_loss += enc_loss.item()
            total_dec_loss += dec_loss.item()
            total_vq_loss += vq_loss.item()
            _, predicted = torch.max(classification_outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * total_correct / total
        self.logger.info(
            f'Train Loss - Encoder: {total_enc_loss / len(train_loader):.4f}, Decoder: {total_dec_loss / len(train_loader):.4f}, VQ Loss: {total_vq_loss / len(train_loader):.4f}, Perplexity: {perplexity:.2f}, Accuracy: {accuracy:.2f}%')
        self.test(test_loader)

    def generate_z_new_interpolation(self, z1, z2, lambda_val):
        return lambda_val * z2 + (1 - lambda_val) * z1

