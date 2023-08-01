#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset1
from net1  import DPNet
import time

class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg    = Dataset.Config(datapath=path, snapshot='./out/model1ssim', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()
        self.net.load_state_dict(torch.load(self.cfg.snapshot))

    def show(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image, mask = image.cuda().float(), mask.cuda().float()
                out1u, out2u, out2r, out3r, out4r, out5r = self.net(image)
                # out = out5r


                plt.figure()
                plt.imshow(np.uint8(image[0].permute(1,2,0).cpu().numpy()*self.cfg.std + self.cfg.mean))
                plt.figure()
                plt.imshow(mask[0].cpu().numpy())
                # plt.figure()
                # plt.imshow(out[0, 0].cpu().numpy())
                plt.figure()
                plt.imshow(torch.sigmoid(out5r[0, 0]).cpu().numpy())

                plt.figure()
                plt.imshow(out4r[0, 0].cpu().numpy())
                plt.figure()
                plt.imshow(torch.sigmoid(out4r[0, 0]).cpu().numpy())
                plt.figure()
                plt.imshow(torch.sigmoid(out3r[0, 0]).cpu().numpy())
                plt.figure()
                plt.imshow(torch.sigmoid(out2r[0, 0]).cpu().numpy())
                plt.figure()
                plt.imshow(torch.sigmoid(out2u[0, 0]).cpu().numpy())
                plt.figure()
                plt.imshow(torch.sigmoid(out1u[0, 0]).cpu().numpy())



                plt.pause(0.5)

                plt.ioff()
                plt.show()

    
    def save(self):

        with torch.no_grad():
            total_time = 0

            for image, mask, shape, name in self.loader:
                image = image.cuda().float()

                start_time = time.time()
                _, out2u, _, _, _, _ = self.net(image, shape)
                torch.cuda.synchronize()
                end_time = time.time()
                total_time += end_time - start_time

                pred  = (torch.sigmoid(out2u[0,0])*255).cpu().numpy()
                # pred = np.squeeze(out2u[0,0]).cpu().data.numpy()
                head  = '../out/1ssim/'+ self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0]+'.png',  np.round(pred))
                # print(name[0]+'.png')
            print('Total time：{}'.format(total_time))

if __name__=='__main__':
    
    root = './data/'
    
    for path in ['test','DUT-OMRON']:
    # for path in ['test']:

        t = Test(dataset1, DPNet, root + path)
        t.save()
        # t.show()
