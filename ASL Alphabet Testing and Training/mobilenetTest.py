from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from torchvision import datasets

import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

from torchvision.models import mobilenet_v2
batch_size = 64
number_of_labels = 37

def imageshow(img):
    img = img.cpu()
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transformations = transforms.Compose([
    #transforms.ToPILImage(),
    #transforms.RandomCrop(200, pad_if_needed=True),
    #transforms.RandomHorizontalFlip,
    transforms.Resize(50),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder("C:\\Users\\Elliot Kwilinski\\Downloads\\archive(6)\\Gesture Image Data", transform=transformations)
classes = test_data.classes
print(classes)
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

    score = 0
    for i in range(batch_size):
        if classes[predicted[i]] == classes[labels[i]]:
            score+=1

    print(100 * score / batch_size)


if __name__ == "__main__":
    # Let's load the model we just created and test the accuracy per label
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=37)
    print(model.classifier)
    if torch.cuda.is_available():
        model.cuda()
    path = "MobileASL.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()
