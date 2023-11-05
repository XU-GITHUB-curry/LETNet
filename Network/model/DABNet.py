
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from IPython import embed
from model.transformer import TransBlock
from model.patch import reverse_patches

__all__ = ["DABNet"]


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


def Channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class DABModule(nn.Module):
    def __init__(self, nIn, d=1, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 4, 1, 1, padding=0, bn_acti=True)


        self.dconv3x1 = Conv(nIn // 4, nIn // 4, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 4, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 4, nIn // 4, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 4, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 4, nIn // 4, (dkSize, 1), 1,
                              padding=(1*d, 0), dilation=(d, 1), groups=nIn // 4, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 4, nIn // 4, (1, dkSize), 1,
                              padding=(0, 1*d), dilation=(1, d), groups=nIn // 4, bn_acti=True)

        self.ddconv5x1 = Conv(nIn // 4, nIn // 4, (5, 1), 1,
                              padding=(2*d, 0), dilation=(d, 1), groups=nIn // 4, bn_acti=True)
        self.ddconv1x5 = Conv(nIn // 4, nIn // 4, (1, 5), 1,
                              padding=(0, 2*d), dilation=(1, d), groups=nIn // 4, bn_acti=True)

        self.ddconv7x1 = Conv(nIn // 4, nIn // 4, (7, 1), 1,
                              padding=(3*d, 0), dilation=(d, 1), groups=nIn // 4, bn_acti=True)
        self.ddconv1x7 = Conv(nIn // 4, nIn // 4, (1, 7), 1,
                              padding=(0, 3*d), dilation=(1, d), groups=nIn // 4, bn_acti=True)

        self.bn_relu_2 = BNPReLU(nIn // 4)

        self.conv1x1 = Conv(nIn // 4, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)


        Out3 = self.dconv3x1(output)
        Out3 = self.dconv1x3(Out3)


        Out3_d = self.ddconv3x1(output)
        Out3_d = self.ddconv1x3(Out3_d)


        Out5_d = self.ddconv5x1(output)
        Out5_d = self.ddconv1x5(Out5_d)


        Out7_d = self.ddconv7x1(output)
        Out7_d = self.ddconv1x7(Out7_d)


        out = Out3 + Out3_d + Out5_d + Out7_d

        out = self.bn_relu_2(out)

        out = self.conv1x1(out)

        out = out + input

        out = Channel_shuffle(out, 4)

        return out


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output

class UpsampleingBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output



class DABNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6, block_3 = 6, block_4 = 3):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.bn_prelu_1 = BNPReLU(32)

        # DAB Block 1
        dilation_block_1 = [2, 2, 2]
        self.downsample_1 = DownSamplingBlock(32, 128)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(128, d=dilation_block_1[i]))
        self.bn_prelu_2 = BNPReLU(128)

        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlock(128, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(128)

        # transformer branch
        self.trans_conv = nn.Sequential(
            Conv(3, 3, 3, 2, padding=1, bn_acti=True),
            Conv(3, 3, 3, 2, padding=1, bn_acti=True),
            Conv(3, 3, 3, 2, padding=1, bn_acti=True),
        )
        self.transformer1 = TransBlock(27)
        self.transformer2 = TransBlock(27)
        self.upsample_t1 = UpsampleingBlock(3, 16)
        self.upsample_t2 = UpsampleingBlock(16, 64)

# DECODER
        dilation_block_3 = [4, 4, 8, 8, 16, 16]
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(128, d=dilation_block_3[i]))
        self.upsample_1 = UpsampleingBlock(128, 64)
        self.bn_prelu_4 = BNPReLU(64)


        dilation_block_4 = [2, 2, 2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
            self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                        DABModule(64, d=dilation_block_4[i]))
        self.upsample_2 = UpsampleingBlock(64, 16)
        self.bn_prelu_5 = BNPReLU(16)


        self.classifier = nn.Sequential(Conv(16, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)

        output0_cat = self.bn_prelu_1(output0)

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(output2)


#transformer branch
        out_tran = self.trans_conv(input)
        b, c, h, w = out_tran.shape

        out_tran = self.transformer1(out_tran)
        out_tran = out_tran.permute(0, 2, 1)
        out_tran = reverse_patches(out_tran, (h, w), (3, 3), 1, 1)

        out_tran = self.transformer2(out_tran)
        out_tran = out_tran.permute(0, 2, 1)
        out_tran = reverse_patches(out_tran, (h, w), (3, 3), 1, 1)

        out_tran = self.upsample_t1(out_tran)

#Decoder

        # DAB Block 3
        output3_0 = self.DAB_Block_3(output2_cat)
        output3 = self.upsample_1(output3_0)
        output3_cat = self.bn_prelu_4(output3 + out_tran)

        # DAB Block 4
        output4_0 = self.DAB_Block_4(output3_cat)
        output4 = self.upsample_2(output4_0)
        output4_cat = self.bn_prelu_5(output4)


        out = self.classifier(output4_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DABNet(classes=19).to(device)
    summary(model,(3,512,1024))
