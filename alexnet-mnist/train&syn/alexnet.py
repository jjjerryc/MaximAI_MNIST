import torch
import torch.nn as nn
import ai8x

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=1, dimensions=(28, 28), planes=60, pool=2, fc_inputs=12, bias=False, **kwargs):
        super().__init__()
        assert planes + num_channels <= ai8x.dev.WEIGHT_INPUTS
        assert planes + fc_inputs <= ai8x.dev.WEIGHT_DEPTH-1

        dim = dimensions[0]

        self.conv1 = ai8x.FusedConv2dReLU(num_channels, planes, 3,
                                          padding=1, bias=bias, **kwargs)
        pad = 2 if dim == 28 else 1
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(planes, planes, 3, pool_size=2, pool_stride=2,
                                                 padding=pad, bias=bias, **kwargs)
        dim //= 2 
        if pad == 2:
            dim += 2

        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(planes, 128-planes-fc_inputs, 3,
                                                 pool_size=2, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= 2

        self.conv4 = ai8x.FusedConv2dReLU(128-planes-fc_inputs, 128-planes-fc_inputs, 3,
                                          padding=1, bias=bias, **kwargs)
        dim //= 2
        
        self.conv5 = ai8x.FusedAvgPoolConv2dReLU(128-planes-fc_inputs,
                                                 fc_inputs, 3,
                                                 pool_size=pool, pool_stride=2, padding=1,
                                                 bias=bias, **kwargs)
        dim //= pool
        self.fc1 = ai8x.Linear(fc_inputs*dim*dim, num_classes, bias=True, wide=True, **kwargs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        return x

def alexnet(pretrained=False, **kwargs):
    assert not pretrained
    return AlexNet(**kwargs)
    
models = [
    {
        'name': 'alexnet',
        'min_input': 1,
        'dim': 2,
    }]