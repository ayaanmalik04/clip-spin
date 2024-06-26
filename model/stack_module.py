from __future__ import print_function

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SPIN import spin
import cv2

affine_par = True

class BasicResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(BasicResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, group=1):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class HourglassModuleMTL(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(HourglassModuleMTL, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual1(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(4):
                res.append(self._make_residual1(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual1(block, num_blocks, planes))
                res.append(self._make_residual1(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        rows = x.size(2)
        cols = x.size(3)

        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2_1, low2_2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2_1 = self.hg[n - 1][4](low1)
            low2_2 = self.hg[n - 1][5](low1)
        low3_1 = self.hg[n - 1][2](low2_1)
        low3_2 = self.hg[n - 1][3](low2_2)
        up2_1 = self.upsample(low3_1)
        up2_2 = self.upsample(low3_2)
        out_1 = up1 + up2_1[:, :, :rows, :cols]
        out_2 = up1 + up2_2[:, :, :rows, :cols]

        return out_1, out_2

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

################################################################
############### StackHOurGlassNet with SPIN ################
################################################################
# We added Dual GCN module after the first downsampling layer and only at segmentation branch
# Added Dual GCN at multiple locations (256 x 256 scale) and (128 x 128 scale) 
class StackHourglassNetMTL_DGCNv4(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        task2_classes=37,
        block=BasicResnetBlock,
        in_channels=3,
        num_stacks=2,
        num_blocks=1,
        hg_num_blocks=3,
        depth=3,
    ):
        super(StackHourglassNetMTL_DGCNv4, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, num_blocks)
        self.layer3 = self._make_residual(block, self.num_feats, num_blocks)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg = []
        res_1, fc_1, score_1, _fc_1, _score_1 = [], [], [], [], []
        res_2, fc_2, score_2, _fc_2, _score_2 = [], [], [], [], []

        for i in range(num_stacks):
            hg.append(HourglassModuleMTL(block, hg_num_blocks, self.num_feats, depth))

            res_1.append(self._make_residual(block, self.num_feats, hg_num_blocks))
            res_2.append(self._make_residual(block, self.num_feats, hg_num_blocks))

            fc_1.append(self._make_fc(ch, ch))
            fc_2.append(self._make_fc(ch, ch))

            score_1.append(nn.Conv2d(ch, task1_classes, kernel_size=1, bias=True))
            score_2.append(nn.Conv2d(ch, task2_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                _fc_1.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _fc_2.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                _score_1.append(nn.Conv2d(task1_classes, ch, kernel_size=1, bias=True))
                _score_2.append(nn.Conv2d(task2_classes, ch, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res_1 = nn.ModuleList(res_1)
        self.fc_1 = nn.ModuleList(fc_1)
        self.score_1 = nn.ModuleList(score_1)
        self._fc_1 = nn.ModuleList(_fc_1)
        self._score_1 = nn.ModuleList(_score_1)

        self.res_2 = nn.ModuleList(res_2)
        self.fc_2 = nn.ModuleList(fc_2)
        self.score_2 = nn.ModuleList(score_2)
        self._fc_2 = nn.ModuleList(_fc_2)
        self._score_2 = nn.ModuleList(_score_2)

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(
            self.inplanes, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)
        
        # Final Classifier
        self.angle_decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.angle_decoder1_score = nn.Conv2d(
            self.inplanes, task2_classes, kernel_size=1, bias=True
        )
        self.angle_finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.angle_finalrelu1 = nn.ReLU(inplace=True)
        self.angle_finalconv2 = nn.Conv2d(32, 32, 3)
        self.angle_finalrelu2 = nn.ReLU(inplace=True)
        self.angle_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

        #Spin module
        self.dgcn_seg_l41 = spin(planes=32, ratio=1)
        self.dgcn_seg_l42 = spin(planes=32, ratio=1)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out_1 = []
        out_2 = []

        rows = x.size(2)
        cols = x.size(3)
        # from transformers import CLIPImageProcessor, CLIPVisionModel

        # feature_extractor = CLIPImageProcessor(do_rescale=False).from_pretrained('openai/clip-vit-base-patch32')
        # vm = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        # # vm.base_model.layers = vm.base_model.layers[:-1]
        # print(CLIPVisionModel.)
        # x = feature_extractor(x, return_tensors='pt')
        # x = vm(**x)
        # print(x.shape)
        # print(x.shape)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.layer1(x)
        # x = self.maxpool(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        from transformers import AutoProcessor, CLIPVisionModel

        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        x = torch.clamp(x, min=-71.30953979492188, max=182.69046020507812)
        x = (x - x.min()) / (x.max() - x.min())
        x = processor(images=x, do_rescale=False, return_tensors="pt")
        with torch.no_grad():
            x = model(**x)
            x = x.last_hidden_state
        x = x[:, 1:, :]
        batch_size, num_tokens, feature_dim = x.size()
        grid_size = int(num_tokens ** 0.5)
        assert grid_size * grid_size == num_tokens

        reshaped_output = x.view(batch_size, grid_size, grid_size, feature_dim)
        reshaped_output = reshaped_output.permute(0, 3, 1, 2)

        class UpsampleConv(nn.Module):
            def __init__(self):
                super(UpsampleConv, self).__init__()
                self.conv = nn.ConvTranspose2d(in_channels=768, out_channels=128, kernel_size=3, stride=10, padding=0, output_padding=1)

            def forward(self, x):
                return self.conv(x)

        upsample_model = UpsampleConv()
        x = upsample_model(reshaped_output)
        print(x.shape)
        for i in range(self.num_stacks):
            y1, y2 = self.hg[i](x)
            y1, y2 = self.res_1[i](y1), self.res_2[i](y2)
            y1, y2 = self.fc_1[i](y1), self.fc_2[i](y2)
            score1, score2 = self.score_1[i](y1), self.score_2[i](y2)
            out_1.append(
                score1[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
            )
            out_2.append(
                score2[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
            )
            if i < self.num_stacks - 1:
                _fc_1, _fc_2 = self._fc_1[i](y1), self._fc_2[i](y2)
                _score_1, _score_2 = self._score_1[i](score1), self._score_2[i](score2)
                x = x + _fc_1 + _score_1 + _fc_2 + _score_2

        # Final Classifications
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ] #d1 = 128, 128,128
        d1_score = self.decoder1_score(d1)
        out_1.append(d1_score)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f2 = self.dgcn_seg_l41(f2) #Graph reasoning at LEVEL 4 - 257 x 257 - SPIN layer 1
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        #f4 = self.dgcn_seg_l42(f4) #Graph reasoning at LEVEL 4 - 257 x 257 - SPIN layer 2
        f5 = self.finalconv3(f4)
        out_1.append(f5)

        # Final Classification
        a_d1 = self.angle_decoder1(y2)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        a_d1_score = self.angle_decoder1_score(a_d1)
        out_2.append(a_d1_score)
        a_f1 = self.angle_finaldeconv1(a_d1)
        a_f2 = self.angle_finalrelu1(a_f1)
        a_f3 = self.angle_finalconv2(a_f2)
        a_f4 = self.angle_finalrelu2(a_f3)
        a_f5 = self.angle_finalconv3(a_f4)
        out_2.append(a_f5)

        return out_1, out_2