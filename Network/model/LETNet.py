import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from IPython import embed
from module.transformer import TransBlock
from module.patch import reverse_patches

__all__ = ["LETNet"]


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


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv1x1_in = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=False)
        self.conv3x1 = Conv(nIn // 2, nIn // 2, (kSize, 1), 1, padding=(1, 0), bn_acti=True)
        self.conv1x3 = Conv(nIn // 2, nIn // 2, (1, kSize), 1, padding=(0, 1), bn_acti=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ca11 = eca_layer(nIn // 2)
        
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca22 = eca_layer(nIn // 2)

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
        self.shuffle = ShuffleBlock(nIn // 2)
        
    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_in(output)
        output = self.conv3x1(output)
        output = self.conv1x3(output)
        
        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br1 = self.ca11(br1)
        
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        br2 = self.ca22(br2)

        output = br1 + br2 + output
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        output = self.shuffle(output + input)

        return output

        #return output + input

#class DABModule(nn.Module):
#    def __init__(self, nIn, d=1, kSize=3, dkSize=3):  #
#        super().__init__()
        #
#        self.bn_relu_1 = BNPReLU(nIn)  #

#        self.conv1x1_init = Conv(nIn, nIn // 2, 1, 1, padding=0, bn_acti=True)  #
#        self.ca0 = eca_layer(nIn // 2)
#        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)
#        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)

#        self.dconv1x3_l = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1), groups=nIn // 2, bn_acti=True)
#        self.dconv3x1_l = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1, 0), groups=nIn // 2, bn_acti=True)

#        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
#        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
#        self.ddconv1x3_r = Conv(nIn // 2, nIn // 2, (1, dkSize), 1, padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
#        self.ddconv3x1_r = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)

#        self.bn_relu_2 = BNPReLU(nIn // 2)
#        self.ca11 = eca_layer(nIn // 2)
#        self.ca22 = eca_layer(nIn // 2)
#        self.ca = eca_layer(nIn // 2)
#        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)
#        self.shuffle_end = ShuffleBlock(groups=nIn // 2)

#    def forward(self, input):
#        output = self.bn_relu_1(input)
#        output = self.conv1x1_init(output)

#        br1 = self.dconv3x1(output)
#        br1 = self.dconv1x3(br1)
#        b1 = self.ca11(br1)


#        br2 = self.ddconv3x1(output)
#        br2 = self.ddconv1x3(br2)
#        b2 = self.ca22(br2)


#        output = self.ca0(output)+ b1 + b2

#        output = self.bn_relu_2(output)

#        output = self.conv1x1(output)
#        output = self.ca(output)
#        out = self.shuffle_end(output + input)
#        return out

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
    
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
        
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ContextBlock(nn.Module):
    def __init__(self,inplanes,ratio,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out    
        
class LongConnection(nn.Module):
    def __init__(self, nIn, nOut, kSize,  bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti
        self.dconv3x1 = nn.Conv2d(nIn, nIn // 2, (kSize, 1), 1, padding=(1, 0))
        self.dconv1x3 = nn.Conv2d(nIn // 2, nOut, (1, kSize), 1, padding=(0, 1))
        
        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.dconv3x1(input)
        output = self.dconv1x3(output)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
                 

class LETNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=12, block_3=12, block_4=3, block_5 = 3, block_6 = 3):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 2, padding=1, bn_acti=True),
        )

        self.bn_prelu_1 = BNPReLU(32)

        self.downsample_1 = DownSamplingBlock(32, 64)

        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(64)

        # DAB Block 2
        dilation_block_2 = [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32]
        self.downsample_2 = DownSamplingBlock(64, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(128)

        # DAB Block 3
        #dilation_block_3 = [2, 5, 7, 9, 13, 17]
        dilation_block_3 = [1,1, 2, 2, 4, 4, 8, 8, 16, 16,32,32]
        self.downsample_3 = DownSamplingBlock(128, 32)
        self.DAB_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.DAB_Block_3.add_module("DAB_Module_3_" + str(i),
                                        DABModule(32, d=dilation_block_3[i]))
        self.bn_prelu_4 = BNPReLU(32)
        
        

        self.transformer1 = TransBlock(dim=288)
        
        
#DECODER
        dilation_block_4 = [2, 2, 2]
        self.DAB_Block_4 = nn.Sequential()
        for i in range(0, block_4):
           self.DAB_Block_4.add_module("DAB_Module_4_" + str(i),
                                       DABModule(32, d=dilation_block_4[i]))
        self.upsample_1 = UpsampleingBlock(32, 16)
        self.bn_prelu_5 = BNPReLU(16)
        

        dilation_block_5 = [2, 2, 2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        DABModule(16, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(16, 16)
        self.bn_prelu_6 = BNPReLU(16)
        
        dilation_block_5 = [2, 2, 2]
        self.DAB_Block_5 = nn.Sequential()
        for i in range(0, block_5):
            self.DAB_Block_5.add_module("DAB_Module_5_" + str(i),
                                        DABModule(16, d=dilation_block_5[i]))
        self.upsample_2 = UpsampleingBlock(16, 16)
        self.bn_prelu_6 = BNPReLU(16)
        
        
        dilation_block_6 = [2, 2, 2]
        self.DAB_Block_6 = nn.Sequential()
        for i in range(0, block_6):
            self.DAB_Block_6.add_module("DAB_Module_6_" + str(i),
                                        DABModule(16, d=dilation_block_6[i]))
        self.upsample_3 = UpsampleingBlock(16, 16)
        self.bn_prelu_7 = BNPReLU(16)
        
        
        self.PA1 = PA(16)
        self.PA2 = PA(16)
        self.PA3 = PA(16)


        
        self.LC1 = LongConnection(64, 16, 3)
        self.LC2 = LongConnection(128, 16, 3)
        self.LC3 = LongConnection(32, 16, 3)
        
        self.classifier = nn.Sequential(Conv(16, classes, 1, 1, padding=0))

    def forward(self, input):

        output0 = self.init_conv(input)
        output0 = self.bn_prelu_1(output0)

        # DAB Block 1
        output1_0 = self.downsample_1(output0)
        output1 = self.DAB_Block_1(output1_0)
        output1 = self.bn_prelu_2(output1)

        # DAB Block 2
        output2_0 = self.downsample_2(output1)
        output2 = self.DAB_Block_2(output2_0)
        output2 = self.bn_prelu_3(output2)

        # DAB Block 3
        output3_0 = self.downsample_3(output2)
        output3 = self.DAB_Block_3(output3_0)
        output3 = self.bn_prelu_4(output3)

#Transformer

        b, c, h, w = output3.shape
        output4 = self.transformer1(output3)
        
        output4 = output4.permute(0, 2, 1)
        output4 = reverse_patches(output4, (h, w), (3, 3), 1, 1)
        
#DECODER            
        output4 = self.DAB_Block_4(output4)
        output4 = self.upsample_1(output4 + self.LC3(output3))
        
        output4 = self.bn_prelu_5(output4)
        
        
        output5 = self.DAB_Block_5(output4)
        output5 = self.upsample_2(output5 + self.LC2(output2))
        
        output5 = self.bn_prelu_6(output5)
        
        
        output6 = self.DAB_Block_6(output5)
        output6 = self.upsample_3(output6 + self.LC1(output1))
        output6 = self.PA3(output6)
        output6 = self.bn_prelu_7(output6)
        
        
        out = F.interpolate(output6, input.size()[2:], mode='bilinear', align_corners=False)
        out = self.classifier(out)
        return out


"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LETNet(classes=19).to(device)
    summary(model, (3, 512, 1024))


