import torch
import torch.nn as nn
import torch.nn.functional as F



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
        
        self.conv_layer_1=nn.Conv2d(1,64,kernel_size=3)

        self.conv_layer_2=nn.Conv2d(64,64,kernel_size=3)

        self.conv_layer_3=nn.Conv2d(64,128,kernel_size=3)
        
        self.conv_layer_4=nn.Conv2d(128,128,kernel_size=3)
        
        self.conv_layer_5=nn.Conv2d(128,256,kernel_size=3)
        
        self.conv_layer_6=nn.Conv2d(256,256,kernel_size=3)

        self.conv_layer_7=nn.Conv2d(256,512,kernel_size=3)
        
        self.conv_layer_8=nn.Conv2d(512,512,kernel_size=3)
        
        self.conv_layer_9=nn.Conv2d(512,1024,kernel_size=3)
        
        self.conv_layer_10=nn.Conv2d(1024,1024,kernel_size=3)

              


    def forward(self,x):
        print(x.size())
        x1=self.conv_layer_1(x)
        print("After 1st Conv :", x1.size())
        x2=F.relu(x1)
        print(x2.size())
        x3=self.conv_layer_2(x2)
        print("After 2nd Conv :", x3.size())
        x4=F.relu(x3)
        print(x4.size())
        x5=self.maxpool(x4)
        print("After Double Conv :",x5.size())
    
        return x5
        #x2=self.maxpool(x1)
        #x3=self.conv_layer_2(x2)
        #x4=self.maxpool(x3)
        #x5=self.conv_layer_3(x4)
        #x6=self.maxpool(x5)
        #x7=self.conv_layer_4(x6)
        #x8=self.maxpool(x7)
        #x9=self.conv_layer_5(x8)
        #print(x9.size())

        
if __name__=="__main__":
    image=torch.rand((1,1,572,572))
    model=Unet()
    model(image)
    #print(image)
    




        
        






