## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class NaimishNet(nn.Module):

    def __init__(self):
        super(NaimishNet, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # Formula:     (W − F + 2*P ) / S + 1
        
        # Following the Naimishnet Architecture we need:
        # 4 covolution2d layers, 4 maxpooling2d layers and 3 dense layers, with "sandwiched" dropout and
        # activation layers, as shown in table
        
        #layer 2
        self.conv1 = nn.Conv2d(1, 32, 4)

        #layer 3 
        self.activation1 = nn.ELU()
        
        #layer 4 
        self.pool1 = nn.MaxPool2d(2, 2)
               
        #layer 5 dropout1
        self.drop1 = nn.Dropout(0.1)
        
        #layer 6 
        self.conv2 = nn.Conv2d(32, 64, 3)#output= 110-3/1 +1=108
        
        #layer 7 
        self.activation2 = nn.ELU()
        
        #layer 8 
        self.pool2 = nn.MaxPool2d(2, 2)
        
        #layer 9
        self.drop2 = nn.Dropout(0.2)
        
        #layer 10
        self.conv3 = nn.Conv2d(64, 128, 2)
        
        #layer 11
        self.activation3 = nn.ELU()
        
        #layer 12 
        self.pool3 = nn.MaxPool2d(2, 2)
        
        #layer 13
        self.drop3 = nn.Dropout(0.3)
        
        #layer 14
        self.conv4 = nn.Conv2d(128, 256, 1)
            
        #layer 15
        self.activation4 = nn.ELU()
        
        #layer 16
        self.pool4 = nn.MaxPool2d(2, 2)
        
        #layer 17
        self.drop4 = nn.Dropout(0.4)
        
        #layer 18
        self.flatten1 = Flatten()
        
        #layer 19 input = 256 *13*13
        self.fc1 = nn.Linear(43264, 512)
        
        #layer 20 
        self.activation5 = nn.ELU()
        
        #layer 21
        self.drop5 = nn.Dropout(0.5)
        
        #layer 22
        self.fc2 = nn.Linear(512, 256)
                
        #layer 23
        self.activation6 = nn.ELU()
       
        #layer 24
        self.drop6 = nn.Dropout(0.6)
        
        #layer 25
        self.fc3 = nn.Linear(256, 136)
        
#         # the points for the prediction without training become more spread
#         nn.init.xavier_uniform_(self.fc1.weight.data)
#         nn.init.xavier_uniform_(self.fc2.weight.data)
#         nn.init.xavier_uniform_(self.fc3.weight.data)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #layers 2 to 5 conv1
        #print("input shape:",x.shape)
        x = self.conv1(x)
        #print("conv1 shape:",x.shape)
        x = self.activation1(x)
        #print("activation1 shape:",x.shape)
        x = self.pool1(x)
        #print("pool1 shape:",x.shape)
        x = self.drop1(x)
        #print("drop1 shape:",x.shape)
        
        #layers 6 to 9 conv2
        x = self.drop2(self.pool2(self.activation2(self.conv2(x))))
        #print("input drop2:",x.shape)
        #layers 10 to 13 conv3
        x = self.drop3(self.pool3(self.activation3(self.conv3(x))))
        #print("input drop3:",x.shape)
        
        #layers 14 to 17 conv4
        
        #print("input shape:",x.shape)
        x = self.conv4(x)
        #print("conv4 shape:",x.shape)
        x = self.activation4(x)
        #print("activation4 shape:",x.shape)
        x = self.pool4(x)
        #print("pool4 shape:",x.shape)
        x = self.drop4(x)
        #print("drop4 shape:",x.shape)
        
        #x = self.drop4(self.pool4(self.activation4(self.conv4(x))))
        #print("input drop4:",x.shape)
        #layers 18 to 21 dense1
        x = self.drop5(self.activation5(self.fc1(self.flatten1(x))))
        
        #layers 22 to 24 dense2
        #print("drop5 shape:",x.shape)
        x = self.fc2(x)
        #print("fc2 shape:",x.shape)
        x = self.activation6(x)
        #print("activation6 shape:",x.shape)
        x = self.drop6(x)
        #print("drop6 shape:",x.shape)
        #x = self.drop6(self.activation6(self.fc2(x)))
        # final layer 25
        x = self.fc3(x)
        #print("Dense3 shape:",x.shape)        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
