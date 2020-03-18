import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

resnext = torchvision.models.resnext50_32x4d(pretrained=True)

for param in resnext.parameters():
    param.requires_grad = False

resnext.fc = nn.Linear(resnext.fc.in_features,100) # 输入尺度为resnext fc层的输入尺度，输出为100

images = torch.randn(64,3,224,224)
outputs = resnext(images)
print(outputs.size())

torch.save(resnext, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnext.state_dict(), 'params.ckpt')
resnext.load_state_dict(torch.load('params.ckpt'))