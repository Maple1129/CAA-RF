import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import numpy as np
from collections import Counter

'''600轮收敛完成69%'''

# 加载CSV文件的数据集
data_file_path = rf'.\Data_set\dataset_sum - 副本.csv'
data = pd.read_csv(data_file_path, header=None)

# 分离特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用ADASYN处理数据不平衡
adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Get the output of the last time step
        return out

global epoch_order,LSTM_time_order,Train_Accuracy,LSTM_Test_Accuracy,LSTM_F1_accuracy
epoch_order = []
LSTM_time_order = [0.0]
Train_Accuracy = []
LSTM_Test_Accuracy = []
LSTM_F1_accuracy = []

# Model parameters
input_size = X_train.shape[1]  # 输入特征的数量
hidden_size = 100
num_layers = 2
num_classes = len(torch.unique(y_train_tensor))  # 类别数量

# Initialize model, loss function, and optimizer
model = RNNModel(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 70000
for epoch in range(num_epochs):

    start = time.time()
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    duration = time.time() - start
    LSTM_time_order.append(duration + LSTM_time_order[-1])
    epoch_order.append(epoch+1)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
    # Evaluate on training set (optional but useful for debugging)
    train_outputs = model(X_train_tensor)
    _, train_predicted = torch.max(train_outputs, 1)
    train_accuracy = (train_predicted == y_train_tensor).float().mean()
    Train_Accuracy.append(train_accuracy)
    print(f'Train Accuracy: {train_accuracy.item():.6f}')

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean()
        accuracy_f1 = f1_score(y_test_tensor, predicted, average='macro')

        LSTM_F1_accuracy.append(accuracy_f1)
        LSTM_Test_Accuracy.append(accuracy)
        print(f'Test Accuracy: {accuracy.item():.6f}')

    with open(r"./result/RNN/epoch_order.txt", "a") as f:
        f.write(f"{epoch},")
    with open(r"./result/RNN/time_order.txt", "a") as f:
        f.write(f"{LSTM_time_order[-1]:.6f},")
    with open(r"./result/RNN/Test_Accuracy.txt", "a") as f:
        f.write(f"{LSTM_Test_Accuracy[-1]:.6f},")
    with open(r"./result/RNN/Train_Accuracy.txt", "a") as f:
        f.write(f"{Train_Accuracy[-1]:.6f},")
    with open(r"./result/RNN/F1_accuracy.txt", "a") as f:
        f.write(f"{accuracy_f1:.6f},")


