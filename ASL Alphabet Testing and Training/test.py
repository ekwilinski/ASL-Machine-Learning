from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import coremltools as ct

batch_size = 16
number_of_labels = 29

def imageshow(img):
    img = img.cpu()
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]) 

test_data = datasets.ImageFolder("C:\\Users\\Elliot Kwilinski\\Downloads\\archive(2)\\asl_alphabet_test\\asl_alphabet_test", transform=transformations)
classes = test_data.classes
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))
    images, labels = images.cuda(), labels.cuda()
    # show all images as one image grid
    print(images.shape)
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=27, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(27)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=27, out_channels=81, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(81)
        self.conv5 = nn.Conv2d(in_channels=81, out_channels=243, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(243)
        self.fc1 = nn.Linear(243*100*100, 29)

    def forward(self, input):

        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))   
        #print(output.shape)  
        output = output.view(-1, 243*100*100)
        output = self.fc1(output)
        
        return output

if __name__ == "__main__":
    # Let's load the model we just created and test the accuracy per label
    model = Network()
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    path = "mySecondModel.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()

    # Trace the model with random data.
    example_input = torch.rand(16,3,200,200) 
    model.cpu()
    traced_model = torch.jit.trace(model, example_input)
    out = traced_model(example_input)

    mlModel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
    )

    mlModel.save("asl_v1.0.mlmodel")