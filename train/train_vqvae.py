import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import logging

class VQVAE_Trainer:
    def __init__(self, model, train_loader, test_loader, recon_weight, vq_weight, optimizer, criterion, device, save_path, logger):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.optimizer = optimizer
        self.criterion = criterion  # MSE Loss for reconstruction
        self.device = device
        self.save_path = save_path
        self.logger = logger

    def save_model(self):
        torch.save(self.model.state_dict(), f'{self.save_path}/vqvae_model.pth')
        self.logger.info("VQVAE model saved successfully.")

    def test(self, test_loader):
        self.model.eval()

        total_dec_loss, total_vq_loss = 0, 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                # Forward pass
                recon_images, encoder_out, vq_loss, perplexity, _ = self.model(images)

                # Compute loss
                loss = self.criterion(recon_images, images)

                total_dec_loss += loss.item()
                #total_vq_loss += vq_loss

        avg_loss = total_dec_loss / len(test_loader)
        #avg_vq_loss = total_vq_loss / len(test_loader)
        self.logger.info(f'Test Reconstruction Loss: {avg_loss:.4f},Test VQ Loss: {vq_loss:.4f},Test perplexity: {perplexity:.4f} ')

    def train(self, epochs):
        self.logger.info("Training VQVAE model")
        scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_dec_loss, total_vq_loss, total_correct, total, total_perplexity = 0, 0, 0, 0, 0

            for images, _ in self.train_loader:
                images = images.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                recon_images, encoder_out, vq_loss, perplexity, _ = self.model(images)

                # Compute reconstruction loss
                dec_loss = self.criterion(recon_images, images)

                all_loss = self.recon_weight * dec_loss + self.vq_weight * vq_loss

                # Backward pass and optimization
                all_loss.backward()
                self.optimizer.step()
                total_dec_loss += dec_loss.item()
                total_vq_loss += vq_loss.item()
                total_loss += all_loss.item()
                total_perplexity += perplexity.item()

            #avg_loss = total_loss / len(self.train_loader)
            scheduler.step()

            self.logger.info(
                f'Epoch {epoch + 1}, Current learning rate: {scheduler.get_last_lr()[0]},  Train Loss - Decoder: {total_dec_loss / len(self.train_loader):.4f}, VQ Loss: {total_vq_loss / len(self.train_loader):.4f}, Perplexity: {total_perplexity/ len(self.train_loader):.4f}%')

            self.test(self.test_loader)

        self.save_model()
