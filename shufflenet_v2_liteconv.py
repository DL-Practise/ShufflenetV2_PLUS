import torch
import torch.nn as nn



def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class LiteConv_V1(nn.Module):
    def __init__(self, channel, gate=1):
        super(LiteConv_V1, self).__init__()
        self.channel = channel
        self.gate = gate

        self.spatial_conv = nn.Sequential(
            self.depthwise_conv(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel))

        self.channel_conv = nn.Sequential(
            nn.Conv2d(channel, gate, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(gate))

        self.gc = nn.Sequential(
            nn.Conv2d(channel * (self.gate+1), channel, kernel_size=3, stride=1, padding=1, groups=channel,bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        x_dw = self.spatial_conv(x)
        x_pw = self.channel_conv(x)

        nB = x_dw.shape[0]
        nDWC = x_dw.shape[1]
        nH = x_dw.shape[2]
        nW = x_dw.shape[3]
        nPWC = x_pw.shape[1]

        x_dw = x_dw.reshape(nB, nDWC, nH * nW)
        x_pw = x_pw.reshape(nB, 1, -1)
        x_pw = x_pw.repeat(1, nDWC, 1)
        x_merge = torch.cat((x_dw, x_pw), dim=2)
        x_merge = x_merge.reshape(nB, nDWC * (self.gate + 1), nH, nW)
        x_out = self.gc(x_merge)
        return x_out

class LiteConv_V2(nn.Module):
    def __init__(self, channel):
        super(LiteConv_V2, self).__init__()
        self.channel = channel
        self.dw1 = nn.Sequential(
            self.depthwise_conv(channel+2, channel+2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel+2))
        self.dw2 = nn.Sequential(
            self.depthwise_conv(channel + 2, channel + 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel+2))
        self.dw3 = nn.Sequential(
            self.depthwise_conv(channel + 2, channel + 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel+2))

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        nB = x.shape[0]
        nH = x.shape[2]
        nW = x.shape[3]
        dev = x.device
        padding = torch.zeros((nB, 1, nH, nW)).to(dev)
        x = torch.cat([padding, x, padding], dim=1)

        x_dw1 = self.dw1(x)
        x_dw2 = self.dw2(x)
        x_dw3 = self.dw2(x)
        x = x_dw1[:,0:self.channel] + x_dw2[:,1:self.channel+1] + x_dw3[:,2:self.channel+2]
        x = nn.functional.relu(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            #nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(branch_features),
            #nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class ShuffleNetV2LiteConv(nn.Module):
    def __init__(self, arg_dict):
        super(ShuffleNetV2LiteConv, self).__init__()

        num_classes = arg_dict['class_num']
        width_mult = arg_dict['channel_ratio']
        inverted_residual=InvertedResidual
		
        if width_mult == 0.5:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 244, 488, 976, 2048]
        elif width_mult == 0.25:
            stages_repeats = [4, 8, 4]
            stages_out_channels = [24, 28, 48, 96, 512]
        else:
            assert(False)
		

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Linear(output_channels, num_classes)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self._forward_impl(x)
        if self.training:
            return x
        else:
            return torch.softmax(x, dim=1)
		
	
    def train_step(self, images, labels):
        out = self.forward(images)
        loss = nn.functional.cross_entropy(out, labels)
        return loss

    def eval_step(self, images):
        out = self.forward(images)
        return out

