#!/usr/bin/python3
# coding=utf-8

import sys
import datetime

# sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import dataset
from net import MSRMNet
import os
import random
import numpy as np
from thop import profile
from thop import clever_format

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ssim_loss = ssim.SSIM(window_size=11, size_average=True)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    ssim_out = 1 - ssim_loss(pred, mask)

    return (wbce + wiou+0.1*ssim_out).mean()


def train(Dataset, Network):
    ## dataset
    cfg = Dataset.Config(datapath='./data/train', savepath='./exp/run', mode='train', batch=20, lr=0.00005, momen=0.9,#0.00005
                         decay=5e-4, epoch=180)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8,
                        drop_last=False, pin_memory=True)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)

    ## val dataloader
    val_cfg = Dataset.Config(datapath='./data/test', mode='test')
    val_data = Dataset.Data(val_cfg)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)
    min_mae = 1.0
    best_epoch = 0
    ## network
    net = Network(cfg)
    net.train(True)

    x = torch.rand(1, 3, 384, 384)
    flops, params = profile(net, inputs=(x,))
    # print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)

    net.cuda()


    seed = 7
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## parameter
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter:{}".format(total))

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.00001)
    if torch.cuda.device_count() > 1:
        # torch.distributed.init_process_group(backend="nccl")
        # net = nn.parallel.DistributedDataParallel(net)  # use multiple GPU
        net = nn.DataParallel(net)

    # sw = SummaryWriter(cfg.savepath)

    for epoch in range(cfg.epoch):
        global_step = 0
        for step, (image, mask) in enumerate(loader):
            image, mask= image.cuda().float(), mask.cuda().float()
            out1u, out2u, out2r, out3r, out4r, out5r = net(image)

            loss1u = structure_loss(out1u, mask)
            loss2u = structure_loss(out2u, mask)

            loss2r = structure_loss(out2r, mask)
            loss3r = structure_loss(out3r, mask)
            loss4r = structure_loss(out4r, mask)
            loss5r = structure_loss(out5r, mask)

            loss = (loss1u + loss2u) / 2 + loss2r / 2 + loss3r / 4 + loss4r / 8 + loss5r / 16

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            if step % 100 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f' % (
                datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                loss.item()))

        # sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch + 1)
        # sw.add_scalars('loss', {'loss1u': loss1u.item(), 'loss2u': loss2u.item(), 'loss2r': loss2r.item(),
        #                         'loss3r': loss3r.item(), 'loss4r': loss4r.item(), 'loss5r': loss_edge.item()},
        #                global_step=epoch + 1)
        scheduler.step()
        if epoch > cfg.epoch / 2:
            mae = validate(net, val_loader, 5019)
            print('duts MAE:%s' % mae)
            if mae < min_mae:
                min_mae = mae
                best_epoch = epoch + 1
                torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1) + str(mae)[0:7])
            print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))


def validate(model, val_loader, nums):
    model.train(False)
    avg_mae = 0.0

    # fLog = open(pa + '/' + str(ep) + 'mae' + '.txt', 'w')

    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            _, out, _, _, _, _ = model(image)
            pred = torch.sigmoid(out[0, 0] * 255)
            temp = torch.abs(pred - mask[0]).mean()
            # fLog.write(str(temp.item()) + '\n')
            avg_mae += temp
            # avg_mae += torch.abs(pred - mask[0]).mean()
        # fLog.close()

    model.train(True)
    return (avg_mae / nums).item()


if __name__ == '__main__':
    train(dataset, MSRMNet)
