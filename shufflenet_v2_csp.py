import torch
import torch.nn as nn



__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


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
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
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


class ShuffleNetV2CSP(nn.Module):
    def __init__(self, arg_dict):
        super(ShuffleNetV2CSP, self).__init__()

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
            downsamping_block = inverted_residual(input_channels, output_channels, 2)
            setattr(self, name + '_down', downsamping_block)
            seq = []
            internel_channels = int(output_channels / 2)
            for i in range(repeats - 1):
                seq.append(inverted_residual(internel_channels, internel_channels, 1))
            setattr(self, name, nn.Sequential(*seq))

            #csp_conv1 = nn.Sequential(nn.Conv2d(internel_channels, internel_channels, 1, bias=False),
            #                          nn.BatchNorm2d(internel_channels),
            #                          nn.ReLU(inplace=True))
            csp_conv2 = nn.Sequential(nn.Conv2d(internel_channels, internel_channels, 1, bias=False),
                                      nn.BatchNorm2d(internel_channels),
                                      nn.ReLU(inplace=True))
            #csp_conv3 = nn.Sequential(nn.Conv2d(output_channels, output_channels, 1, bias=False),
            #                          nn.BatchNorm2d(output_channels),
            #                          nn.ReLU(inplace=True))
            #setattr(self, name + '_csp_conv1', csp_conv1)
            setattr(self, name + '_csp_conv2', csp_conv2)
            #setattr(self, name + '_csp_conv3', csp_conv3)

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

        s2_donw = self.stage2_down(x)
        s2_ori = self.stage2(s2_donw[:, :int(s2_donw.shape[1] / 2), :, :])
        #s2_ori = self.stage2_csp_conv1(s2_ori)
        s2_csp = self.stage2_csp_conv2(s2_donw[:, int(s2_donw.shape[1] / 2):, :, :])
        s2_cat = torch.cat((s2_ori, s2_csp), dim=1)
        #s2 = self.stage2_csp_conv3(s2_cat)
        s2 = channel_shuffle(s2_cat, 2)

        s3_donw = self.stage3_down(s2)
        s3_ori = self.stage3(s3_donw[:, :int(s3_donw.shape[1] / 2), :, :])
        #s3_ori = self.stage3_csp_conv1(s3_ori)
        s3_csp = self.stage3_csp_conv2(s3_donw[:, int(s3_donw.shape[1] / 2):, :, :])
        s3_cat = torch.cat((s3_ori, s3_csp), dim=1)
        #s3 = self.stage3_csp_conv3(s3_cat)
        s3 = channel_shuffle(s3_cat, 2)

        s4_donw = self.stage4_down(s3)
        s4_ori = self.stage4(s4_donw[:, :int(s4_donw.shape[1] / 2), :, :])
        #s4_ori = self.stage4_csp_conv1(s4_ori)
        s4_csp = self.stage4_csp_conv2(s4_donw[:, int(s4_donw.shape[1] / 2):, :, :])
        s4_cat = torch.cat((s4_ori, s4_csp), dim=1)
        #s4 = self.stage4_csp_conv3(s4_cat)
        s4 = channel_shuffle(s4_cat, 2)

        x = self.conv5(s4)
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

