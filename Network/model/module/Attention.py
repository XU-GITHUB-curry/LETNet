import torch
from torch import nn


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c , _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale
        y = torch.mul(x, y)
        return y


class GCBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(GCBlock, self).__init__()
        self.channel = channel
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=2)

        self.conv_mask = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(channel, channel//r, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LayerNorm([self.channel//r, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//r, channel, kernel_size=1, stride=1, padding=0, bias=False),
        )


    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        input_x = x
        out = x
        # [N, C, H * W]
        input_x = input_x.view(b,c,h*w)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(b, 1, h*w)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(b, c, 1, 1)

        channel_add_term = self.channel_add_conv(context)
        out = out + channel_add_term

        return out


# class GCBlock(nn.Module):
#     def __init__(self, channel, r=16):
#         super(GCBlock, self).__init__()
#         self.channel = channel
#         self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
#                                   bias=False)
#         self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
#                                     bias=False)
#         self.conv_g = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
#                                 bias=False)
#         self.softmax = nn.Softmax(dim=2)
#         self.conv_mask = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
#                                    bias=False)
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, channel // r, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // r, channel, kernel_size=1, stride=1, padding=0, bias=False),
#         )
#
#     def forward(self, x):
#         # [N, C, H , W]
#         b, c, h, w = x.size()
#         # [N, C, H*W]
#         x_phi = self.conv_phi(x).view(b, c, -1)
#         # [N, H * W, 1]
#         x_theta = self.conv_theta(x).view(b, 1, -1).permute(0, 2, 1).contiguous()
#         # # [N, H * W, 1, 1]
#         # x_theta = x_theta.view(b, h*w, 1, 1)
#         x_theta = self.softmax(x_theta)
#         # [N, C, 1, 1]
#         mul_theta_phi = torch.matmul(x_phi, x_theta)
#         # [N, C, 1, 1]
#         mul_theta_phi = self.fc(mul_theta_phi)
#
#         mul_theta_phi = mul_theta_phi.view(b, self.channel, h, w)
#         out = x + mul_theta_phi
#         return out

# if __name__=='__main__':
#     model = NonLocalBlock(channel=32)
#     print(model)
#
#     input = torch.randn(1, 32, 64, 64)
#     out = model(input)
#     print(out.shape)