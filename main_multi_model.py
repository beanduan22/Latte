import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

# ===== project utilities (keep as in your repo) =====
from train.train_en_de import DNN_Model_Trainer
from tools.ModelVisualizer import *
from tools.log import *
from tools.parser import parse_model_path
from evaluate.evaluate import *
from evaluate.evaluate_after_train import *
from train.train_vqvae import VQVAE_Trainer
from tools.merged import *
from tools.load_save_loader import *
from tools.loader_label_matching import *
from model.X_Discriminator import ImageDiscriminator
from model.Z_Discriminator import LatentDiscriminator
from train.train_xd_zd_generator import AdversarialTraining

# ===== your existing DNNs =====
from model.Model import Lenet4_Model_mnist, Lenet5_Model_mnist
from model.Model import Vgg16_Model_cifar10, Resnet18_Model_cifar10
from model.Model import Resnet50_Model_imagenet, Vgg19_Model_imagenet
# newly added in your repo (from prior step)
from model.Model import AllCNNA_Model_cifar10, AllCNNB_Model_cifar10
from model.Model import Custom_Model_1_6B, Custom_Model_3_3B

# ===== your VQ-VAEs =====
from model.VQVAE import Lenet4_VQVAE_mnist, Lenet5_VQVAE_mnist
from model.VQVAE import Vgg16_VQVAE_cifar10, Resnet18_VQVAE_cifar10
from model.VQVAE import Vgg19_VQVAE_imagenet, Resnet50_VQVAE_imagenet
# newly added in your repo (from prior step)
from model.VQVAE import Lenet5_VQVAE_fashionmnist, Resnet18_VQVAE_svhn

os.environ['OMP_NUM_THREADS'] = '1'

# =========================
# Config
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# choose one of: "mnist", "fashionmnist", "svhn"
DATASET = "mnist"

# general training params
train_batch_size = 64
test_batch_size  = 256
pre_train_epoch = 10
vqvae_train_epoch = 30

# losses / opt
learning_rate = 1e-3
dec_loss_weight_ = 2.0
vq_loss_weight_  = 1.0

# adversarial (optional, can be disabled)
USE_ADVERSARIAL_PHASE = False
inter_lambda = 0.4
adv_epoch = 10
adv_enc_loss_weight = 1.0
adv_dec_loss_weight = 10.0
adv_vq_loss_weight = 1.0
adv_lr_encoder = 1e-3
adv_lr_zd = 1e-3
adv_lr_xd = 1e-3

# =========================
# Data loaders
# =========================
def prepare_data_mnist():
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, 10

def prepare_data_fashionmnist():
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, 10

def prepare_data_svhn():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, 10

# =========================
# Multi-model setup per dataset
# =========================
def build_models_for(dataset_key, num_classes):
    """
    Returns:
      (model_a, model_b, vqvae_model, save_path)
    """
    if dataset_key == "mnist":
        model_a = Lenet4_Model_mnist().to(DEVICE)
        model_b = Lenet5_Model_mnist().to(DEVICE)
        vqvae  = Lenet5_VQVAE_mnist(in_channel=1).to(DEVICE)
        save_path = parse_model_path("mnist_lenet4_lenet5_multimodel")
    elif dataset_key == "fashionmnist":
        # names are labels; tune configs inside class if you need the exact capacity
        model_a = Custom_Model_1_6B(num_classes=num_classes).to(DEVICE)
        model_b = Custom_Model_3_3B(num_classes=num_classes).to(DEVICE)
        vqvae  = Lenet5_VQVAE_fashionmnist(in_channel=1).to(DEVICE)
        save_path = parse_model_path("fashionmnist_custom_multimodel")
    elif dataset_key == "svhn":
        model_a = AllCNNA_Model_cifar10(num_classes=num_classes).to(DEVICE)
        model_b = AllCNNB_Model_cifar10(num_classes=num_classes).to(DEVICE)
        vqvae  = Resnet18_VQVAE_svhn(in_channel=3).to(DEVICE)
        save_path = parse_model_path("svhn_allcnn_multimodel")
    else:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    return model_a, model_b, vqvae, save_path

# =========================
# Train helpers
# =========================
def train_dnn(model, train_loader, val_loader, save_path, logger, epochs=pre_train_epoch):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    trainer = DNN_Model_Trainer(model, train_loader, val_loader, optimizer, criterion, DEVICE, save_path=save_path, log=logger)
    trainer.train(epochs)

def train_vqvae(model, train_loader, val_loader, save_path, logger, epochs=vqvae_train_epoch):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    trainer = VQVAE_Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        recon_weight=dec_loss_weight_,
        vq_weight=vq_loss_weight_,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        save_path=save_path,
        logger=logger
    )
    trainer.train(epochs=epochs)

# =========================
# Differential tester
# =========================
@torch.no_grad()
def evaluate_disagreement(model_a, model_b, data_loader, device, logger, max_log=5):
    model_a.eval(); model_b.eval()
    total = 0
    disagree = 0
    samples_logged = 0

    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[0].to(device), batch[1].to(device)
        else:
            # SVHN returns a tuple; ensure compatibility if dataloader differs
            x, y = batch

        logits_a = model_a(x)
        logits_b = model_b(x)
        pred_a = logits_a.argmax(dim=1)
        pred_b = logits_b.argmax(dim=1)

        mism = (pred_a != pred_b)
        disagree += mism.sum().item()
        total += x.size(0)

        # optionally log a few example indices
        if logger is not None and samples_logged < max_log:
            idxs = torch.nonzero(mism).flatten().tolist()
            for i in idxs:
                if samples_logged >= max_log:
                    break
                logger.info(f"[DIFF] idx={i}, predA={pred_a[i].item()}, predB={pred_b[i].item()}, label={y[i].item()}")
                samples_logged += 1

    rate = (disagree / total) if total > 0 else 0.0
    if logger is not None:
        logger.info(f"Disagreement: {disagree}/{total} = {rate:.4%}")
    return rate

# =========================
# Main
# =========================
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    # logger
    log_manager = LogManager()
    logger = log_manager.get_logger()
    logger.info("Starting multi-model pipeline...")

    # data
    if DATASET == "mnist":
        train_loader, val_loader, num_classes = prepare_data_mnist()
    elif DATASET == "fashionmnist":
        train_loader, val_loader, num_classes = prepare_data_fashionmnist()
    elif DATASET == "svhn":
        train_loader, val_loader, num_classes = prepare_data_svhn()
    else:
        raise ValueError("DATASET must be one of: 'mnist', 'fashionmnist', 'svhn'")

    # models
    model_a, model_b, vqvae, save_path = build_models_for(DATASET, num_classes)
    os.makedirs(save_path, exist_ok=True)

    # train both classifiers
    logger.info("Pretraining model A...")
    train_dnn(model_a, train_loader, val_loader, os.path.join(save_path, "modelA"), logger)

    logger.info("Pretraining model B...")
    train_dnn(model_b, train_loader, val_loader, os.path.join(save_path, "modelB"), logger)

    # train VQ-VAE (shared generator for test-case synthesis)
    logger.info("Training VQ-VAE...")
    train_vqvae(vqvae, train_loader, val_loader, os.path.join(save_path, "vqvae"), logger)

    # (Optional) adversarial phase using generator to synthesize inputs
    if USE_ADVERSARIAL_PHASE:
        logger.info("Starting adversarial generator phase...")
        xd = ImageDiscriminator().to(DEVICE)
        zd = LatentDiscriminator().to(DEVICE)
        adv_training = AdversarialTraining(
            dnnmodel=model_a,  # you can run separate phases or swap to model_b as needed
            vqvae=vqvae,
            zd=zd,
            xd=xd,
            save_path=os.path.join(save_path, "adv"),
            logger=logger,
            enc_loss_weight=adv_enc_loss_weight,
            dec_loss_weight=adv_dec_loss_weight,
            vq_loss_weight=adv_vq_loss_weight,
            lr_encoder=adv_lr_encoder,
            lr_zd=adv_lr_zd,
            lr_xd=adv_lr_xd
        )
        # provide appropriate loaders; here we just reuse train/val for demo
        z_adv_loader = adv_training.train(
            loader_1=train_loader,
            loader_2=val_loader,
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=adv_epoch,
            lambda_=inter_lambda,
            epsilon=0.01,
            delta=0.1
        )
        # You can also run differential testing on synthesized loader if it yields (x,y)
        # evaluate_disagreement(model_a, model_b, z_adv_loader, DEVICE, logger)

    # Differential testing on validation set
    logger.info("Running differential test (model A vs model B) on validation set...")
    diff_rate = evaluate_disagreement(model_a, model_b, val_loader, DEVICE, logger)

    logger.info(f"[{DATASET}] Differential disagreement rate: {diff_rate:.4%}")
    logger.info("Program finished successfully.")
