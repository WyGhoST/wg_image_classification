import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

#1. Tải tập huấn luyện
train_dataset = MNIST(
    root = 'data',
    train = True,
    download = True,
    transform=ToTensor()   
)

#2. Tải tập kiểm tra
test_dataset = MNIST(
    root='data',
    train=False,
    download = True,
    transform= ToTensor()
)

#3. Kiểm tra số lượng mẫu
print('Số lượng ảnh train: ', len(train_dataset))
print('Số lượng ảnh text: ',len(test_dataset))

#4. Kiểm một mẫu 
image, label = train_dataset[0]

print('********* One sample ***********')
print('Shape ảnh: ',image.shape)
print('Nhãn: ', label)
print('Kiểu dữ liệu ảnh: ',image.dtype)
print('Giá trị pixel nhỏ nhất: ', image.min().item())
print('Giá trị pixel lớn nhất: ', image.max().item())

#5. Gom dữ liệu thành các batch
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True
)

#Lấy batch đầu
images, labels = next(iter(train_loader))

print('****** One batch *****')
print('Shape của images: ', images.shape)
print('Shape của labels: ', labels.shape)
print('16 nhãn đầu tiên: ', labels[:16])

#6. Hiển thị 16 ảnh đầu tiên trong batch
fig, axes = plt.subplots(4,4,figsize=(4,4))

for image, label, ax in zip(
    images[:16],
    labels[:16],
    axes.flatten()
):
    ax.imshow(image.squeeze(0), cmap = 'grey')
    
    ax.set_title(f'Label:{label.item()}')
    ax.axis('off')
    
plt.tight_layout()
plt.show()