import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        # 编码器5层
        self.enconv1 = nn.Sequential(
            self._block(in_channels, 64, 3, 1),
            self._block(64, 64, 3, 1),
        )
        self.enconv2 = nn.Sequential(
            self._block(64, 128, 3, 1),
            self._block(128, 128, 3, 1),
            self._block(128, 128, 3, 1),
        )
        self.enconv3 = nn.Sequential(
            self._block(128, 256, 3, 1),
            self._block(256, 256, 3, 1),
            self._block(256, 256, 3, 1),
        )
        self.enconv4 = nn.Sequential(
            self._block(256, 512, 3, 1),
            self._block(512, 512, 3, 1),
            self._block(512, 512, 3, 1),
        )
        self.enconv5 = nn.Sequential(
            self._block(512, 512, 3, 1),
            self._block(512, 512, 3, 1),
            self._block(512, 512, 3, 1),
        )

        # 解码器
        self.deconv1 = nn.Sequential(
            self._block(512, 512, 3, 1),
            self._block(512, 512, 3, 1),
            self._block(512, 512, 3, 1),
        )
        self.deconv2 = nn.Sequential(
            self._block(512, 512, 3, 1),
            self._block(512, 512, 3, 1),
            self._block(512, 256, 3, 1),
        )
        self.deconv3 = nn.Sequential(
            self._block(256, 256, 3, 1),
            self._block(256, 256, 3, 1),
            self._block(256, 128, 3, 1),
        )
        self.deconv4 = nn.Sequential(
            self._block(128, 128, 3, 1),
            self._block(128, 128, 3, 1),
            self._block(128, 64, 3, 1),
        )
        self.deconv5 = nn.Sequential(
            self._block(64, 64, 3, 1),
            self._block(64, 64, 3, 1),
            self._block(64, num_classes, 3, 1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.enconv1(x)
        # 重要特征提取
        out, idx1 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv2(out)
        out, idx2 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv3(out)
        out, idx3 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv4(out)
        out, idx4 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)
        out = self.enconv5(out)
        out, idx5 = F.max_pool2d(out, kernel_size=2, stride=2, return_indices=True)

        # 解码器还原
        out = F.max_unpool2d(out, indices=idx5, kernel_size=2, stride=2, padding=0)
        out = self.deconv1(out)
        out = F.max_unpool2d(out, indices=idx4, kernel_size=2, stride=2, padding=0)
        out = self.deconv2(out)
        out = F.max_unpool2d(out, indices=idx3, kernel_size=2, stride=2, padding=0)
        out = self.deconv3(out)
        out = F.max_unpool2d(out, indices=idx2, kernel_size=2, stride=2, padding=0)
        out = self.deconv4(out)
        out = F.max_unpool2d(out, indices=idx1, kernel_size=2, stride=2, padding=0)
        out = self.deconv5(out)
        # out = self.softmax(out)
        return out




    def _block(self, in_channels, out_channels, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

if __name__ == '__main__':
    net = SegNet(in_channels=3, num_classes=10)
    x = torch.rand(4, 3, 224, 224)
    print(net(x).shape)