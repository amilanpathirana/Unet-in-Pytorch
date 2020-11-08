import torch
import torch.nn as nn
import torch.nn.functional as F








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

        self.uppool1=nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)

        self.conv_layer_11=nn.Conv2d(1024,512,kernel_size=3)

        self.conv_layer_12=nn.Conv2d(512,512,kernel_size=3)

        self.uppool2=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)

        self.conv_layer_13=nn.Conv2d(512,256,kernel_size=3)

        self.conv_layer_14=nn.Conv2d(256,256,kernel_size=3)

        self.uppool3=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)

        
        self.conv_layer_15=nn.Conv2d(256,128,kernel_size=3)

        self.conv_layer_16=nn.Conv2d(128,128,kernel_size=3)

        self.uppool4=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
              

        self.conv_layer_17=nn.Conv2d(128,64,kernel_size=3)

        self.conv_layer_18=nn.Conv2d(64,64,kernel_size=3)

        self.conv_layer_19=nn.Conv2d(64,2,kernel_size=1)

        
              

    def forward(self,x):
        print("Input Size To 1st Layer :",x.size())
        x1=self.conv_layer_1(x)
        x2=self.relu(x1)
        x3=self.conv_layer_2(x2)
        print("After Duble Conolution :", x3.size())
        x4=self.relu(x3)
        print("Size of X4 going to the right side :", x4.size())
        x5=self.maxpool(x4)
        print("After MaxPool :",x5.size())
        print("\n")
    
        print("Input Size To 2nd Layer :",x5.size())
        x6=self.conv_layer_3(x5)
        x7=self.relu(x6)
        x8=self.conv_layer_4(x7)
        print("After Duble Conolution :", x8.size())
        x9=self.relu(x8)
        print("Size of X9 going to the right side :", x9.size())
        x10=self.maxpool(x9)
        print("After MaxPool :",x10.size())
        print("\n")

        print("Input Size To 3rd Layer :",x10.size())
        x11=self.conv_layer_5(x10)
        x12=self.relu(x11)
        x13=self.conv_layer_6(x12)
        print("After Duble Conolution :", x13.size())
        x14=self.relu(x13)
        print("Size of X14 going to the right side :", x14.size())
        x15=self.maxpool(x14)
        print("After MaxPool :",x15.size())
        print("\n")


        print("Input Size To 4th Layer :",x15.size())
        x16=self.conv_layer_7(x15)
        x17=self.relu(x16)
        x18=self.conv_layer_8(x17)
        print("After Duble Conolution :", x18.size())
        x19=self.relu(x18)
        print("Size of X19 going to the right side :", x19.size())
        x20=self.maxpool(x19)
        print("After MaxPool :",x20.size())


        print("\n")
        print("Input Size To 5th Layer :",x20.size())
        x21=self.conv_layer_9(x20)
        x22=self.relu(x21)
        x23=self.conv_layer_10(x22)
        print("After Duble Conolution :", x23.size())
        x24=self.relu(x23)
        x25=self.uppool1(x24)
        print("After UpPool :",x25.size())
        
                
        print("\n")
        sizeoftensortobecropped=x19.shape[2]
        sizeoftargettensor=x25.shape[2]
        print("Width of the left side tensor : ",sizeoftensortobecropped)
        print("Width of the right side tensor : ",sizeoftargettensor)
        delta=sizeoftensortobecropped-sizeoftargettensor
        x19cropped=x19[:,:,delta//2:sizeoftensortobecropped-delta//2,delta//2:sizeoftensortobecropped-delta//2]
        print("Width of the left side tensor after cropping : ", x19cropped.shape[2])


        print("\n")
        x26=torch.cat([x19cropped,x25],1)
        print("After Cat :",x26.size())

        print("\n")
        print("Input Size To 6th Layer :",x26.size())
        x27=self.conv_layer_11(x26)
        x28=self.relu(x27)
        x29=self.conv_layer_12(x28)
        print("After Duble Conolution :", x29.size())
        x30=self.relu(x29)
        x31=self.uppool2(x30)
        print("After UpPool :",x31.size())


        print("\n")
        sizeoftensortobecropped=x14.shape[2]
        sizeoftargettensor=x31.shape[2]
        print("Width of the left side tensor : ",sizeoftensortobecropped)
        print("Width of the right side tensor : ",sizeoftargettensor)
        delta=sizeoftensortobecropped-sizeoftargettensor
        x14cropped=x14[:,:,delta//2:sizeoftensortobecropped-delta//2,delta//2:sizeoftensortobecropped-delta//2]
        print("Width of the left side tensor after cropping : ", x14cropped.shape[2])


        print("\n")
        x32=torch.cat([x14cropped,x31],1)
        print("After Cat :",x32.size())

        print("\n")
        print("Input Size To 7th Layer :",x32.size())
        x33=self.conv_layer_13(x32)
        x34=self.relu(x33)
        x35=self.conv_layer_14(x34)
        print("After Duble Conolution :", x35.size())
        x36=self.relu(x35)
        x37=self.uppool3(x36)
        print("After UpPool :",x37.size())

        print("\n")
        sizeoftensortobecropped=x9.shape[2]
        sizeoftargettensor=x37.shape[2]
        print("Width of the left side tensor : ",sizeoftensortobecropped)
        print("Width of the right side tensor : ",sizeoftargettensor)
        delta=sizeoftensortobecropped-sizeoftargettensor
        x9cropped=x9[:,:,delta//2:sizeoftensortobecropped-delta//2,delta//2:sizeoftensortobecropped-delta//2]
        print("Width of the left side tensor after cropping : ", x9cropped.shape[2])


        print("\n")
        x38=torch.cat([x9cropped,x37],1)
        print("After Cat :",x38.size())


        print("\n")
        print("Input Size To 8th Layer :",x38.size())
        x39=self.conv_layer_15(x38)
        x40=self.relu(x39)
        x41=self.conv_layer_16(x40)
        print("After Duble Conolution :", x41.size())
        x42=self.relu(x41)
        x43=self.uppool4(x42)
        print("After UpPool :",x43.size())



        print("\n")
        sizeoftensortobecropped=x4.shape[2]
        sizeoftargettensor=x43.shape[2]
        print("Width of the left side tensor : ",sizeoftensortobecropped)
        print("Width of the right side tensor : ",sizeoftargettensor)
        delta=sizeoftensortobecropped-sizeoftargettensor
        x4cropped=x4[:,:,delta//2:sizeoftensortobecropped-delta//2,delta//2:sizeoftensortobecropped-delta//2]
        print("Width of the left side tensor after cropping : ", x4cropped.shape[2])


        print("\n")
        x44=torch.cat([x4cropped,x43],1)
        print("After Cat :",x44.size())


        print("\n")
        print("Input Size To 9th Layer :",x44.size())
        x45=self.conv_layer_17(x44)
        x46=self.relu(x45)
        x47=self.conv_layer_18(x46)
        print("After Duble Conolution :", x47.size())
        x48=self.relu(x47)
        x49=self.conv_layer_19(x48)
        print("Final output :", x49.size())
        #x49=self.uppool4(x48)
        #print("After UpPool :",x43.size())




        return x49


        
if __name__=="__main__":
    image=torch.rand(1,1,572,572)
    
    model=Unet()
    model(image)


    




        
        






