from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from segnet_model import SegNet
from collections import Counter

image_path = r"data/voc2011/VOCdevkit/VOC2012/JPEGImages/2007_000392.jpg"

# Load the image
image = Image.open(image_path).convert("RGB")

# Define the transformations to be applied
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

tf = transforms.Compose([
    transforms.Resize((224, 224)),
])

# Apply the transformations to the image
image = transform(image)

mask = Image.open(r"data/voc2011/VOCdevkit/VOC2012/SegmentationClass/2007_000392.png")
mask = tf(mask)
model = SegNet(3, 21)
# Pass the image through the model
model.load_state_dict(torch.load('save_model/SegNet/best_model.pth'))
model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0))

# Convert the output to a numpy array
# output = output.squeeze(0)
# output = output.detach().numpy().round()
# output = np.argmax(output, axis=0)
output = torch.argmax(output.squeeze(), dim=0).detach().cpu()
# output = output.numpy()


# Convert the output to a PIL image
# output = Image.fromarray(np.uint8(output))
# output.show()

mask = np.array(mask)

mask[mask==255] = 0

# output.show()
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(output)
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.show()


