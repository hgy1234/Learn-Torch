import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


image_path="Image/shipe.png"
image=Image.open(image_path)

transform=torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32,32)),
     torchvision.transforms.ToTensor()
     ]
)
image=transform(image)
image=torch.reshape(image,(1,3,32,32))

class f_module(nn.Module):
    def __init__(self):
        super(f_module, self).__init__()
        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

#加载已经训练好的模型
model=torch.load("f_m9.pth")
# print(model)
#由于模型是GPU训练的，因此需要做如下转换

image=image.cuda()
model=model.cuda()
output=model(image)

print(output)
print(output.argmax())