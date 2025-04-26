import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

# Tạo CustomDataset
class ParityDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randint(0, 100, (size,))  # Số nguyên ngẫu nhiên từ 0-99
        self.labels = (self.data % 2).long()  # 0: chẵn, 1: lẻ
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].float(), self.labels[idx]

# Tạo dataset 
dataset = ParityDataset(1000)

# Chia train/test
indices = np.arange(len(dataset))
np.random.shuffle(indices)
train_indices = indices[:800]
test_indices = indices[800:]
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Định nghĩa mô hình
class ParityMLP(nn.Module):
    def __init__(self):
        super(ParityMLP, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # 2 lớp: chẵn/lẻ
    
    def forward(self, x):
        x = x.view(-1, 1)  # Reshape thành [batch_size, 1]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Khởi tạo mô hình, loss, optimizer
model = ParityMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện
model.train()
for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Đánh giá
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
