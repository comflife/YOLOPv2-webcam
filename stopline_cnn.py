import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 1. 랜덤한 정지선 데이터 생성
# 1. 랜덤한 정지선 데이터 생성
def generate_data(num_samples=1000):
    data = np.zeros((num_samples, 256, 256))
    labels = np.zeros(num_samples)

    for i in range(num_samples):
        row = np.random.randint(0, 256)
        start_col = np.random.randint(0, 256)
        length = np.random.randint(256//3, 2*256//3)
        end_col = start_col + length

        if end_col > 255:
            end_col = 255

        data[i, row, start_col:end_col] = 1
        labels[i] = 1  # 정지선

    return data, labels


data, labels = generate_data()

# 2. 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 모델 학습
data_tensor = torch.FloatTensor(data).unsqueeze(1)
labels_tensor = torch.LongTensor(labels)

for epoch in range(20):  # loop over the dataset multiple times
    optimizer.zero_grad()
    outputs = model(data_tensor)
    loss = criterion(outputs, labels_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 모델 저장
torch.save(model.state_dict(), "stop_line_detection_model.pth")

print("Model saved as stop_line_detection_model.pth")