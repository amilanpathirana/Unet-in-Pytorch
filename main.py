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

        self.relu=nn.ReLU()
        
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
        print("Input Size To 1st Layer :",x.size())
        x1=self.conv_layer_1(x)
        x2=self.relu(x1)
        x3=self.conv_layer_2(x2)
        print("After Duble Conolution :", x3.size())
        x4=self.relu(x3)
        x5=self.maxpool(x4)
        print("After First MaxPool :",x5.size())
        print("\n")
    
        print("Input Size To 2nd Layer :",x5.size())
        x6=self.conv_layer_3(x5)
        x7=self.relu(x6)
        x8=self.conv_layer_4(x7)
        print("After Duble Conolution :", x8.size())
        x9=self.relu(x8)
        x10=self.maxpool(x9)
        print("After First MaxPool :",x10.size())
        print("\n")

        print("Input Size To 3rd Layer :",x10.size())
        x11=self.conv_layer_5(x10)
        x12=self.relu(x11)
        x13=self.conv_layer_6(x12)
        print("After Duble Conolution :", x13.size())
        x14=self.relu(x13)
        x15=self.maxpool(x14)
        print("After First MaxPool :",x15.size())
        print("\n")


        print("Input Size To 4th Layer :",x15.size())
        x16=self.conv_layer_7(x15)
        x17=self.relu(x16)
        x18=self.conv_layer_8(x17)
        print("After Duble Conolution :", x18.size())
        x19=self.relu(x18)
        x20=self.maxpool(x19)
        print("After First MaxPool :",x20.size())
        print("\n")


        print("Input Size To 4th Layer :",x20.size())
        x21=self.conv_layer_9(x20)
        x22=self.relu(x21)
        x23=self.conv_layer_10(x22)
        print("After Duble Conolution :", x23.size())
        x24=self.relu(x23)
        x25=self.maxpool(x24)
        print("After First MaxPool :",x25.size())
        print("\n")

        return x5


        
if __name__=="__main__":
    image=torch.rand((1,1,572,572))
    model=Unet()
    model(image)

    




        
        






