#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2
from swin import SwinTransformer
from resnet_encoder import ResNet

import sys

sys.path.append('./')


class Res2NetBlock(nn.Module):
    def __init__(self, outplanes, inchannel, scales=2):
        super(Res2NetBlock, self).__init__()

        self.scales = scales
        # 1*1的卷积层
        self.inconv = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.BatchNorm2d(outplanes),
        )

        # 3*3的卷积层，一共有3个卷积层和3个BN层
        self.conv1 = nn.Sequential(
            nn.Conv2d(outplanes // 2, 64, 1),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes // 2, 64, 1),
            nn.Conv2d(64, 64, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(64),
        )

        # 1*1的卷积层
        self.outconv = nn.Sequential(
            nn.Conv2d(inchannel, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
        )
        self.outconv2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
        )

    def forward(self, x):
        input = x
        x = self.inconv(x)

        # scales个部分
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        ys.append(F.relu(self.conv1(xs[0])))
        ys.append(F.relu(self.conv2(xs[1]) + ys[0]))

        y = torch.cat(ys, 1)

        y = self.outconv(y)
        input = self.outconv2(input)
        output = F.relu(y+input)

        return output


class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')

        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h + left)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v + down)), inplace=True)
        fuse = out2h * out2v

        out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True) + out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h + fuse)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True) + out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v + fuse)), inplace=True)

        return out4h, out4v

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45 = CFM()
        self.cfm34 = CFM()
        self.cfm23 = CFM()

    def forward(self, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            refine5 = F.interpolate(fback, size=out5v.size()[2:], mode='bilinear')
            refine4 = F.interpolate(fback, size=out4h.size()[2:], mode='bilinear')
            refine3 = F.interpolate(fback, size=out3h.size()[2:], mode='bilinear')
            refine2 = F.interpolate(fback, size=out2h.size()[2:], mode='bilinear')
            out5v = out5v + refine5
            out4h, out4v = self.cfm45(out4h + refine4, out5v)
            out3h, out3v = self.cfm34(out3h + refine3, out4v)
            out2h, pred = self.cfm23(out2h + refine2, out3v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, pred = self.cfm23(out2h, out3v)
        return out2h, out3h, out4h, out5v, pred


class MSRMNet(nn.Module):
    def __init__(self, cfg):
        super(MSRMNet, self).__init__()

        # #resnet
        # self.backbone = ResNet()
        # path = './pvt/resnet50-19c8e357.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)
        #
        # self.squeeze5 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.squeeze4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.squeeze3 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        # self.squeeze2 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())


        # # pvt
        # self.backbone = pvt_v2_b2()
        # path = './pvt/pvt_v2_b2.pth'
        # save_model = torch.load(path)
        # model_dict = self.backbone.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # self.backbone.load_state_dict(model_dict)
        #
        # self.squeeze5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        # self.squeeze4 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        # self.squeeze3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        # self.squeeze2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())

        #swin
        self.backbone = SwinTransformer(img_size=384,
                                       embed_dim=128,
                                       depths=[2, 2, 18, 2],
                                       num_heads=[4, 8, 16, 32],
                                       window_size=12)

        pretrained_dict = torch.load('./pvt/swin_base_patch4_window12_384_22k.pth')["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.backbone.state_dict()}
        self.backbone.load_state_dict(pretrained_dict)


        self.squeeze5 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        self.squeeze4 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        self.squeeze3 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())
        self.squeeze2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),  nn.GroupNorm(32,64), nn.PReLU())

        self.res = Res2NetBlock(64, 128)
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, shape=None):
        pvt = self.backbone(x)

        # out2h = pvt[0]
        # out3h = pvt[1]
        # out4h = pvt[2]
        # out5v = pvt[3]

        #swin
        out2h = pvt[4]
        out3h = pvt[3]
        out4h = pvt[2]
        out5v = pvt[1]



        out2h, out3h, out4h, out5v = self.squeeze2(out2h), self.squeeze3(out3h), self.squeeze4(out4h), self.squeeze5(
            out5v)

        out2h = self.res(out2h)
        out3h = self.res(out3h)
        out4h = self.res(out4h)
        out5v = self.res(out5v)

        out2h, out3h, out4h, out5v, pred1 = self.decoder1(out2h, out3h, out4h, out5v)
        out2h, out3h, out4h, out5v, pred2 = self.decoder2(out2h, out3h, out4h, out5v, pred1)
        pred2 = pred1
        shape = x.size()[2:] if shape is None else shape
        pred1 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear')

        out2h = F.interpolate(self.linearr2(out2h), size=shape, mode='bilinear')
        out3h = F.interpolate(self.linearr3(out3h), size=shape, mode='bilinear')
        out4h = F.interpolate(self.linearr4(out4h), size=shape, mode='bilinear')
        out5h = F.interpolate(self.linearr5(out5v), size=shape, mode='bilinear')

        return pred1, pred2, out2h, out3h, out4h, out5h
