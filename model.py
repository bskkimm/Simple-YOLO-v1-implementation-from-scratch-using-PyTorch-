import torch.nn as nn
import torch
import config


S = config.S # the entire image is divided into SxS(=7x7) cells
B = config.B # a cell predicts up to B(=2) bounding boxes
C = config.C # The number of classes is C(=20)


class Yolo(nn.Module):
    """
    A model reimplementation of the YOLO (You Only Look Once) object detection architecture.

    This model replicates the original paper's architecture as closely as possible.
    As described in the paper, input shape: (Batch, 3, 448, 448), output shape: (Batch, S, S, B*5+20)
    """
    def __init__(self):
        super().__init__()

        self.Conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                                    out_channels=64,
                                                    kernel_size=7,
                                                    stride=2,
                                                    padding=3), # 448 --> 224
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.MaxPool2d(kernel_size=2,
                                                       stride=2)) # 224 --> 112
        self.Conv_block2 = nn.Sequential(nn.Conv2d(64, 
                                                    192,
                                                    kernel_size=3,
                                                    padding=1 # 112 --> 112
                                                    ),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.MaxPool2d(kernel_size=2,
                                                       stride=2)) # 112 -> 56
        self.Conv_block3 = nn.Sequential(nn.Conv2d(192,
                                                    128,
                                                    kernel_size=1, # 56 --> 56
                                                    ),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(128,
                                                    256,
                                                    kernel_size=3,
                                                    padding=1), # 56 --> 56
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(256,
                                                    256,
                                                    kernel_size=1), # 56 --> 56
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(256,
                                                    512,
                                                    kernel_size=3,
                                                    padding=1),  # 56 --> 56
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.MaxPool2d(kernel_size=2,
                                                       stride=2))# 56 --> 28)
        Conv_block4 = []
        for _ in range(4):
            Conv_block4.append(nn.Conv2d(512, 256, kernel_size=1))
            Conv_block4.append(nn.LeakyReLU(0.1, inplace=True))
            Conv_block4.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
            Conv_block4.append(nn.LeakyReLU(0.1, inplace=True))
        
        Conv_block4.append(nn.Conv2d(512, 512, kernel_size=1))
        Conv_block4.append(nn.LeakyReLU(0.1, inplace=True))
        Conv_block4.append(nn.Conv2d(512, 1024, kernel_size=3, padding=1))
        Conv_block4.append(nn.LeakyReLU(0.1, inplace=True))
        Conv_block4.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 28 --> 14
        
        self.Conv_block4 = nn.Sequential(*Conv_block4)

        Conv_block5 = []

        for _ in range(2):
            Conv_block5.append(nn.Conv2d(1024, 512, kernel_size=1))
            Conv_block5.append(nn.LeakyReLU(0.1, inplace=True))
            Conv_block5.append(nn.Conv2d(512, 1024, kernel_size=3, padding=1))
            Conv_block5.append(nn.LeakyReLU(0.1, inplace=True))

        Conv_block5.append(nn.Conv2d(1024, 1024, kernel_size=3, padding=1))
        Conv_block5.append(nn.LeakyReLU(0.1, inplace=True))
        Conv_block5.append(nn.MaxPool2d(kernel_size=2, stride=2)) # 14 --> 7

        self.Conv_block5 = nn.Sequential(*Conv_block5)

        self.Conv_block6 = nn.Sequential(nn.Conv2d(1024,1024, kernel_size=3, padding=1),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Conv2d(1024,1024, kernel_size=3, padding=1),
                                          nn.LeakyReLU(0.1, inplace=True))
        
        self.prediction_head = nn.Sequential(nn.Linear(in_features=1024*7*7, out_features=4096),
                                             nn.LeakyReLU(0.1, inplace=True), 
                                             nn.Dropout(0.5),
                                             nn.Linear(in_features=4096,
                                                       out_features=S*S*(B*5 + C))) 
        
    def forward(self,x):
        x = self.Conv_block1(x)
        x = self.Conv_block2(x)
        x = self.Conv_block3(x)
        x = self.Conv_block4(x)
        x = self.Conv_block5(x)
        x = self.Conv_block6(x)
        x = x.view(x.size(0), -1)
        x = torch.reshape(self.prediction_head(x), #
                          (x.shape[0],S, S, B*5 + C))

        return x
