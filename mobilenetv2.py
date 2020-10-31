class MobileNetBlock(nn.Module):
    def __init__(self, input_channel, t, c, s=1):
        super(MobileNetBlock, self).__init__()
        self.stride = s

        self.skip_connection = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=c, kernel_size=1, stride=self.stride,
                      bias=False),
            nn.BatchNorm2d(c),
        )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels= t * input_channel, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(t * input_channel),
            nn.ReLU6(),
            nn.Conv2d(in_channels=t * input_channel, out_channels=t * input_channel, kernel_size=3, stride=self.stride,
                      bias=False, padding=1, groups=t * input_channel),
            nn.BatchNorm2d(t * input_channel),
            nn.ReLU6(),
            nn.Conv2d(in_channels=t * input_channel, out_channels = c, kernel_size=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        skip_connection = self.skip_connection(x)

        x = self.block(x)
        if self.stride == 1:
            x = skip_connection + x

        return x

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, bias=False)
        self.conv_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)

        self.block1 = MobileNetBlock(32, 1, 16, 1)

        self.block2 = MobileNetBlock(16, 6, 24, 1)

        self.block2_1 = MobileNetBlock(24, 6, 24, 1)

        self.block3 = MobileNetBlock(24, 6, 32, 2)

        self.block3_1 = MobileNetBlock(32, 6, 32, 1)

        self.block4 = MobileNetBlock(32, 6, 64, 2)

        self.block4_1 = MobileNetBlock(64, 6, 64, 1)

        self.block5 = MobileNetBlock(64, 6, 96, 1)

        self.block5_1 = MobileNetBlock(96, 6, 96, 1)

        self.block6 = MobileNetBlock(96, 6, 160, 2)

        self.block6_1 = MobileNetBlock(160, 6, 160, 1)

        self.block7 = MobileNetBlock(160, 6, 320, 1)

        self.block8 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, bias=False)

        self.avgpool = nn.AvgPool2d(kernel_size=7)

        self.final_conv = nn.Conv2d(in_channels=1280, out_channels=100, kernel_size=1, stride=1, bias=False)

        self.classifier = nn.Linear(100, 100)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)

        x = self.conv_1(x)
        x = self.maxpool(x)

        x = self.block1(x)
        #print(b1.shape)

        x = self.block2(x)

        x = self.block2_1(x)
        #print(b2.shape)

        x = self.block3(x)
        x = self.block3_1(x)
        x = self.block3_1(x)
        #print(b3.shape)

        x = self.block4(x)
        x = self.block4_1(x)
        x = self.block4_1(x)
        x = self.block4_1(x)
        #print(b4.shape)

        x = self.block5(x)
        x = self.block5_1(x)
        x = self.block5_1(x)
        #print(b5.shape)

        x = self.block6(x)
        x = self.block6_1(x)
        x = self.block6_1(x)
        #print(b6.shape)

        x = self.block7(x)

        x = self.block8(x)

        features = self.avgpool(x)

        x = self.final_conv(features)
        #print(x.shape)

        x = x.view(features.size(0), -1)
        #print(x.shape)

        x = self.classifier(x)

        return x, features
