import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Tiền xử lý dữ liệu
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten ảnh 28x28 về vector 784
])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# Định nghĩa mô hình MLP
class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Lớp ẩn
        self.relu = nn.ReLU()           # Hàm kích hoạt ReLU
        self.fc2 = nn.Linear(128, 10)   # Lớp đầu ra (10 lớp tương ứng 10 chữ số)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
# Tạo mô hình, hàm mất mát và bộ tối ưu
model = MNIST_MLP()
criterion = nn.CrossEntropyLoss()              # Hàm mất mát cho phân loại đa lớp
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Tối ưu bằng Adam

# Huấn luyện mô hình
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)              # Tính đầu ra
        loss = criterion(output, target) # Tính loss

        optimizer.zero_grad()  # Xoá gradient cũ
        loss.backward()        # Lan truyền ngược gradient
        optimizer.step()       # Cập nhật trọng số

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
