
from train.train_en_de import *
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from tools.ModelVisualizer import *
from tools.log import *
from tools.parser import *
from evaluate.evaluate import *
from evaluate.evaluate_after_train import *
from model.Model import Resnet50_Model_imagenet
from model.Model import Vgg19_Model_imagenet
from model.Model import Resnet18_Model_cifar10
from model.Model import Vgg16_Model_cifar10
from model.Model import Lenet4_Model_mnist
from model.Model import Lenet5_Model_mnist
from model.VQVAE import Resnet18_VQVAE_cifar10
from model.VQVAE import Vgg16_VQVAE_cifar10
from model.VQVAE import Resnet50_VQVAE_imagenet
from model.VQVAE import Vgg19_VQVAE_imagenet
from model.VQVAE import Lenet4_VQVAE_mnist
from model.VQVAE import Lenet5_VQVAE_mnist
import torchvision.transforms as transforms
from train.train_xd_zd_generator import *
from model.X_Discriminator import *
from model.Z_Discriminator import *
from train.train_vqvae import *
from tools.save_load_dataset import *
from tools.loader_label_matching import *
from tools.load_save_loader import *
from tools.data_matching_big import *
from tools.merged import *
import os

os.environ['OMP_NUM_THREADS'] = '1'
label_num = 10
n_per_label = 3333
# pretrain par
pre_train_epoch = 10
vqvae_train_epoch = 50
train_batch_size = 16
test_batch_size = 16
dec_loss_weight_ = 2
vq_loss_weight_ = 1
learning_rate = 0.001
momentum_ = 0.9
# adv train par
inter_lambda = 0.4
adv_epoch = 20
adv_enc_loss_weight = 1.0
adv_dec_loss_weight = 20.0
adv_vq_loss_weight = 1.0
adv_lr_encoder = 0.001
adv_lr_zd = 0.001
adv_lr_xd = 0.001
model_name = 'cifar10_resnet8_encoder_decoder'
# model_name = 'imagenet_resnet50_encoder_decoder'
# model_name = 'imagenet_vgg19_encoder_decoder'
# model_name = 'cifar10_vgg16_encoder_decoder'
# model_name = 'mnist_lenet4_encoder_decoder'
# model_name = 'mnist_lenet5_encoder_decoder'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    torch.manual_seed(torch.initial_seed())  # 重置种子以解除对其他随机操作的影响
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def load_model(save_path, model_name):
    dnn_model, vqvae, xd, zd = parse_model_instance(model_name)
    dnn_model = eval(dnn_model)
    vqvae = eval(vqvae)
    # vqvae = eval(quantizer)
    dnn_model_path = f'{save_path}/dnn_model.pth'
    vqvae_path = f'{save_path}/vqvae_model.pth'

    if os.path.exists(dnn_model_path):
        dnn_model.load_state_dict(torch.load(dnn_model_path))
        logger.info("Loaded dnn model weights from disk.")
    else:
        logger.info("No saved dnn model weights found.")

    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path))
        logger.info("Loaded vqvae model weights from disk.")
    else:
        logger.info("No saved vqvae model weights found.")

    return dnn_model, vqvae


def pre_train(logger, save_path, dnn_model, train_loader, test_loader):
    optimizer = torch.optim.Adam([{"params": dnn_model.parameters(), "lr": learning_rate}])

    criterion = nn.CrossEntropyLoss()
    trainer = DNN_Model_Trainer(dnn_model, train_loader, test_loader, optimizer, criterion,
                                device, save_path=save_path, log=logger)
    trainer.train(pre_train_epoch)


def main_train(logger, save_path, model, train_loader, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_criterion = nn.MSELoss()

    # 创建 VQVAE 训练器
    trainer = VQVAE_Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        recon_weight=dec_loss_weight_,
        vq_weight=vq_loss_weight_,
        optimizer=optimizer,
        criterion=mse_criterion,
        device=device,
        save_path=save_path,
        logger=logger
    )

    # 开始训练
    trainer.train(epochs=vqvae_train_epoch)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    log_manager = LogManager()
    logger = log_manager.get_logger()
    logger.info("Starting the program...")
    save_path = parse_model_path(model_name)
    model, vqvae = load_model(save_path, model_name)
    train_loader, test_loader = prepare_data()
    #pre_train(logger, save_path, model, train_loader, test_loader)
    main_train(logger, save_path, vqvae, train_loader, test_loader)
    pic_dir = save_path + '/compare'

    evaluator = ModelEvaluator(model, vqvae, device, logger, pic_dir)
    # evaluator.evaluate(test_loader)
    loader_1, loader_2 = evaluator.get_input(test_loader)
    merged_dataloader = create_merged_dataloader(loader_1, loader_2, label_num, n_per_label)

    new_loader = select_n_samples_per_class(loader_2, 1, label_num)

    loader_sorted_repeated, loader_matched, for_label_num = prepare_new_dataloaders(new_loader,merged_dataloader,batch_size=16,num_classes=label_num)

    xd = ImageDiscriminator()
    zd = LatentDiscriminator()

    loader_1 = loader_sorted_repeated
    loader_2 = loader_matched

    adv_training = AdversarialTraining(
        dnnmodel=model,
        vqvae=vqvae,
        zd=zd,
        xd=xd,
        save_path=save_path,
        logger=logger,
        enc_loss_weight=adv_enc_loss_weight,
        dec_loss_weight=adv_dec_loss_weight,
        vq_loss_weight=adv_vq_loss_weight,
        lr_encoder=adv_lr_encoder,
        lr_zd=adv_lr_zd,
        lr_xd=adv_lr_xd
    )

    z_adv_loader = adv_training.train(
        loader_1=loader_1,
        loader_2=loader_2,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=adv_epoch,
        lambda_=inter_lambda,
        epsilon=0.01,
        delta=0.1
    )

    model, vqvae = load_model(save_path, model_name)

    evaluator_after_train = ModelEvaluatorAfterTrain(model, vqvae, device, logger, pic_dir)

    evaluator_after_train.latent_compare(z_adv_loader, loader_1)

    evaluator_after_train.evaluate_compare(z_adv_loader, loader_1, for_label_num)

    logger.info("Program finished successfully.")
