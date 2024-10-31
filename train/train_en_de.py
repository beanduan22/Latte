import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import logging
class DNN_Model_Trainer:
    def __init__(self, dnn_model, train_loader, test_loader, optimizer, criterion, device, save_path, log):
        self.dnn_model = dnn_model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.logger = log
    def save_models(self):
        torch.save(self.dnn_model.state_dict(), f'{self.save_path}/dnn_model.pth')
        self.logger.info("DNN Models saved successfully.")

    def test(self, test_loader):
        self.dnn_model.eval()

        total_cls_loss, total_correct, total = 0, 0, 0
        with torch.no_grad():
            total_top5_correct = 0
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Encoder outputs
                classification_outputs = self.dnn_model(images)

                cls_loss = self.criterion(classification_outputs, labels)

                # Log losses and compute accuracy
                total_cls_loss += cls_loss.item()
                _, predicted = torch.max(classification_outputs.data, 1)
                total_correct += (predicted == labels).sum().item()

                _, top5_predicted = classification_outputs.topk(5, 1, True, True)
                top5_correct = top5_predicted.eq(labels.view(-1, 1).expand_as(top5_predicted))
                total_top5_correct += top5_correct.sum().item()

                total += labels.size(0)

        accuracy = 100 * total_correct / total
        accuracy_top5 = 100 * total_top5_correct / total
        self.logger.info(
            f'Test Loss - CLS: {total_cls_loss / len(test_loader):.4f}, Top-1 Accuracy: {accuracy:.2f}%, Top-5 Accuracy: {accuracy_top5:.2f}%')

    def train(self, epochs):
        self.logger.info("training dnn model")
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        #scheduler_1 = torch.optim.lr_scheduler.StepLR(self.optimizer_1, step_size=20, gamma=0.1)
        #scheduler_2 = torch.optim.lr_scheduler.StepLR(self.optimizer_2, step_size=20, gamma=0.1)
        for epoch in range(epochs):
            self.dnn_model.train()
            # T_max 是一个周期的迭代次数
            total_cls_loss, total_correct, total = 0, 0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                classification_outputs = self.dnn_model(images)
                cls_loss = self.criterion(classification_outputs, labels)

                total_loss = cls_loss

                total_loss.backward()
                self.optimizer.step()
                total_cls_loss += cls_loss.item()

                _, predicted = torch.max(classification_outputs.data, 1)
                total_correct += (predicted == labels).sum().item()

                total += labels.size(0)
            accuracy = 100 * total_correct / total
            scheduler.step()
            self.logger.info(
                f'Epoch {epoch+1}, Current learning rate: {scheduler.get_last_lr()[0]},  Train Loss - Encoder: {total_cls_loss / len(self.train_loader):.4f}, Accuracy: {accuracy:.2f}%')
            self.test(self.test_loader)
        self.save_models()

