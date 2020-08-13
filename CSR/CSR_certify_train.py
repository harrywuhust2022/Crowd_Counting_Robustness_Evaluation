import numpy as np
import time
import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
from utils_mean import *
from config import Config
from model import CSRNet
from dataset import create_train_dataloader, create_test_dataloader
from utils import denormalize
from torch.autograd import Variable
import matplotlib.pyplot as plt


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


if __name__ == "__main__":

    keep = 45
    method = 'CSR'
    dataset_name = 'Shanghai_A'

    device = torch.device('cuda:0')
    print('use cuda ==> {}'.format(device))

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')
    output_dir = './saved_models/'

    cfg = Config()  # configuration
    model = CSRNet().to(device)  # model

    criterion = nn.MSELoss(size_average=False)  # objective
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # optimizer
    train_dataloader = create_train_dataloader(cfg.dataset_root, use_flip=True, batch_size=cfg.batch_size)
    test_dataloader = create_test_dataloader(cfg.dataset_root)  # dataloader

    min_mae = sys.maxsize
    min_mae_epoch = -1
    train_loss = 0.0
    for epoch in range(1, cfg.epochs):  # start training
        model.train()
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(train_dataloader)):
            image = data['image'].to(device)
            gt_densitymap = data['densitymap'].to(device)

            image = random_mask_batch_one_sample(image, keep, reuse_noise=True)
            image = Variable(image)
            gt_densitymap = Variable(gt_densitymap)

            et_densitymap = model(image)  # forward propagation
            loss = criterion(et_densitymap, gt_densitymap)  # calculate loss
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()  # back propagation
            optimizer.step()  # update network parameters

        train_loss = epoch_loss/len(train_dataloader)

        if epoch % 100 == 0:
            save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(method, dataset_name, epoch))
            save_net(save_name, model)

        if epoch == 1: # 如果是epoch = 20要改成19
            model.eval()
            with torch.no_grad():
                gt_count = 0.0
                et_count = 0.0
                epoch_mae = 0.0
                mae = 0.0
                mse = 0.0
                correct = 0
                total = 0
                bias = 0

                if not os.path.exists('./results_CSR_ablated'):
                    os.mkdir('./results_CSR_ablated')
                if not os.path.exists('./results_CSR_ablated/density_map_adv'):
                    os.mkdir('./results_CSR_ablated/density_map_adv')
                if not os.path.exists('./results_CSR_ablated/images_adv'):
                    os.mkdir('./results_CSR_ablated/images_adv')
                if not os.path.exists('./results_CSR_ablated/images_gt'):
                    os.mkdir('./results_CSR_ablated/images_gt')

                for i, data in enumerate(tqdm(test_dataloader)):

                    full_imgname = i

                    image = data['image'].to(device)
                    gt_densitymap = data['densitymap'].to(device)

                    image = random_mask_batch_one_sample(image, keep, reuse_noise=False)
                    image = Variable(image)

                    et_densitymap = model(image)           # forward propagation

                    et = et_densitymap.data.detach().cpu().numpy()
                    gt = gt_densitymap.data.detach().cpu().numpy()
                    im_adv = image.data.detach().cpu().numpy()

                    im_adv_save = im_adv[0][0]
                    plt.imsave('./results_CSR_ablated/images_adv/{}'.format(full_imgname), im_adv_save, format='png',
                               cmap=plt.cm.jet)

                    gt_save = gt[0][0]
                    plt.imsave('./results_CSR_ablated/images_gt/{}'.format(full_imgname), gt_save, format='png',
                               cmap=plt.cm.jet)

                    et_save = et[0][0]
                    plt.imsave('./results_CSR_ablated/density_map_adv/{}'.format(full_imgname), et_save, format='png',
                               cmap=plt.cm.jet)

                    et_count = np.sum(et)
                    gt_count = np.sum(gt)

                    mae += abs(gt_count - et_count)
                    mse += (gt_count - et_count) * (gt_count - et_count)

                    bias = abs(gt_count - et_count)
                    if bias < 5:
                        correct += 1
                    total += 1

                accuracy = (correct / total) * 100.0
                print("correct: ", correct)
                print("total: ", total)
                mae = mae / len(test_dataloader)
                mse = np.sqrt(mse / len(test_dataloader))
                print("test_ablated_result: ")
                print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))
                print("ablated_test_accuracy: ", accuracy)

                # 将结果保存到txt
                with open('./Ablated_results.txt', 'a') as file_handle:
                    file_handle.write('Ablated_MAE_:')
                    file_handle.write('\n')
                    file_handle.write(str(mae))
                    file_handle.write('Ablated_MSE_:')
                    file_handle.write('\n')
                    file_handle.write(str(mse))
                    file_handle.write('Ablated_acc_:')
                    file_handle.write('\n')
                    file_handle.write(str(accuracy))
                file_handle.close()

