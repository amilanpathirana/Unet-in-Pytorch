import torch
import torch.nn as nn



def double_conv(in_channels,out_channels):
    conv=nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv




class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv_layer_1=double_conv(1,64)
        self.conv_layer_2=double_conv(64,128)
        self.conv_layer_3=double_conv(128,256)
        self.conv_layer_4=double_conv(256,512)
        self.conv_layer_5=double_conv(512,1024)
               


    def forward(self,x):
        x1=self.conv_layer_1(image)
        print(x1.size())
        x2=self.maxpool(x1)
        x3=self.conv_layer_2(x2)
        x4=self.maxpool(x3)
        x5=self.conv_layer_3(x4)
        x6=self.maxpool(x5)
        x7=self.conv_layer_4(x6)
        x8=self.maxpool(x7)
        x9=self.conv_layer_5(x8)
        print(x9.size())

        
if __name__=="__main__":
    image=torch.rand((1,1,200,200))
    model=Unet()
    model(image)
    print(image)
    




        
        






