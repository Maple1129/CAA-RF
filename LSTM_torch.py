import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pandas as pd
import os
import time
from sklearn.metrics import f1_score


'''完成，上面是准确率'''
global epoch_order,time_order,Train_Accuracy,Test_Accuracy,F1_accuracy
epoch_order = []
time_order = [0.0]
Train_Accuracy = []
Test_Accuracy = []
F1_accuracy = []


# 加载CSV文件的数据集
data_file_path = rf'.\Data_set\dataset_sum.csv'
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = X_train.shape[1]  # 输入特征的数量
hidden_size = 128  # LSTM隐藏层的大小
num_layers = 2  # LSTM层数
num_classes = len(torch.unique(y_train_tensor))  # 类别数量
batch_size = 100
learning_rate = 0.001



# Define the LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

def train_model(model, train_loader, device, epochs, criterion, optimizer, ord):
    model.train()
    for epoch in range(epochs):
        start = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.time()
        duration = end - start
        time_order.append(time_order[-1] + duration)
        epoch_order.append(ord)

        with open(r"./result/LSTM/epoch_order.txt", "a") as f:
            f.write(f"{ord},")
        with open(r"./result/LSTM/time_order.txt", "a") as f:
            f.write(f"{time_order[-1]:.6f},")

        # Calculate accuracy over entire training set after each epoch
        train_correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
        accuracy_train = train_correct / len(train_loader.dataset)
        Train_Accuracy.append(accuracy_train)
        with open(r"./result/LSTM/Train_Accuracy.txt", "a") as f:
            f.write(f"{accuracy_train:.6f},")



# Test the model
def test_model(model, test_loader, device, ord):
    model.eval()  # Set the model to evaluation mode
    all_predictions = []
    all_labels = []
    with torch.no_grad():  # Disable gradient calculation
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy_test = correct / total
            Test_Accuracy.append(accuracy_test)

            # Accumulate predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # After the loop, compute the F1 score
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    F1_accuracy.append(f1_macro)

    with open(r"./result/LSTM/Test_Accuracy.txt", "a") as f:
        f.write(f"{Test_Accuracy[-1]:.6f},")
    with open(r"./result/LSTM/F1_accuracy.txt", "a") as f:
        f.write(f"{F1_accuracy[-1]:.6f},")

    accuracy = 100 * correct / total
    print(f'Test Accuracy of the model on the {total} test samples: {accuracy:.6f}%')
    return accuracy



model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create Data Loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



for epoch in range(7000):
    train_model(model, train_loader, device, 1, criterion, optimizer, ord=epoch+1)
    test_model(model, test_loader, device, ord=epoch+1)
