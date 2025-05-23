import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pandas as pd
import os
import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


'''完成，上面是准确率'''
global epoch_order, LSTM_time_order, Train_Accuracy, LSTM_Test_Accuracy, LSTM_F1_accuracy
global LSTM_RF_time_order, LSTM_RF_Train_Accuracy, LSTM_RF_Test_Accuracy, LSTM_RF_F1_Accuracy
epoch_order = []
LSTM_time_order = [0.0]
Train_Accuracy = []
LSTM_Test_Accuracy = []
LSTM_F1_accuracy = []
LSTM_RF_time_order = [0.0]
LSTM_RF_Train_Accuracy = []
LSTM_RF_Test_Accuracy = []
LSTM_RF_F1_Accuracy = []


# 加载数据
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

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 添加序列维度
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 超参数
input_size = X_train.shape[1]  # 输入特征的数量
hidden_size = 128  # LSTM隐藏层的大小
num_layers = 2  # LSTM层数
num_classes = len(torch.unique(y_train_tensor))  # 类别数量
batch_size = 100
learning_rate = 0.001


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_features=False):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  # 初始化细胞状态

        out, _ = self.lstm(x, (h0, c0))  # LSTM前向传播
        features = out[:, -1, :]  # 获取最后一个时间步的隐藏状态
        
        if return_features:
            return features  # 返回特征
            
        out = self.fc(features)  # 全连接层
        return out

# 训练模型
def train_model(model, train_loader, epochs, criterion, optimizer, ord):
    model.train()
    for epoch in range(epochs):
        start = time.time()
        # 计算整个训练集的准确率
        train_correct = 0
        for inputs, labels in train_loader:

            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end = time.time()
        duration = end - start
        LSTM_time_order.append(LSTM_time_order[-1] + duration)
        epoch_order.append(ord)

        LSTM_train_Accuracy = train_correct / len(train_loader.dataset)
        Train_Accuracy.append(LSTM_train_Accuracy)

        print(f"第{ord}轮次")
        with open(r"./result/LSTM_RF/epoch_order.txt", "a") as f:
            f.write(f"{ord},")
        with open(r"./result/LSTM/epoch_order.txt", "a") as f:
            f.write(f"{ord},")
        with open(r"./result/LSTM/LSTM_time_order.txt", "a") as f:
            f.write(f"{LSTM_time_order[-1]:.6f},")
        with open(r"./result/LSTM/LSTM_train_Accuracy.txt", "a") as f:
            f.write(f"{LSTM_train_Accuracy:.6f},")

# 提取特征
def extract_features(model, data_loader):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            features = model(inputs, return_features=True)
            all_features.append(features.numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)

# 测试模型
def test_model(model, test_loader, ord):
    model.eval()  # 设置模型为评估模式
    all_predictions = []
    all_labels = []
    with torch.no_grad():  # 禁用梯度计算
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy_test = correct / total
            LSTM_Test_Accuracy.append(accuracy_test)

            # 累积预测和标签
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # 计算F1分数
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    LSTM_F1_accuracy.append(f1_macro)

    with open(r"./result/LSTM/LSTM_Test_Accuracy.txt", "a") as f:
        f.write(f"{LSTM_Test_Accuracy[-1]:.6f},")
    with open(r"./result/LSTM/LSTM_F1_accuracy.txt", "a") as f:
        f.write(f"{LSTM_F1_accuracy[-1]:.6f},")

    accuracy = 100 * correct / total
    print(f'Test Accuracy of the model on the {total} test samples: {accuracy:.6f}%')
    return accuracy


model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)  # 使用CPU

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练循环
for epoch in range(7000):
    # 训练LSTM
    train_model(model, train_loader,  1, criterion, optimizer, ord=epoch+1)
    test_model(model, test_loader,  ord=epoch+1)
    
    # 提取特征并训练随机森林
    start_time = time.time()
    
    # 从训练数据中提取特征
    train_features, train_labels = extract_features(model, train_loader)

    start_time = time.time()
    # 训练随机森林
    rf_classifier.fit(train_features, train_labels)
    duration = time.time() - start_time

    # 获取训练数据上的预测
    train_pred = rf_classifier.predict(train_features)
    rf_train_accuracy = accuracy_score(train_labels, train_pred)
    
    # 从测试数据中提取特征并评估
    test_features, test_labels = extract_features(model, test_loader)
    test_pred = rf_classifier.predict(test_features)
    rf_test_accuracy = accuracy_score(test_labels, test_pred)
    rf_f1_score = f1_score(test_labels, test_pred, average='macro')
    

    LSTM_RF_time_order.append(LSTM_time_order[-1] - LSTM_time_order[-2] + duration + LSTM_RF_time_order[-1])
    LSTM_RF_Train_Accuracy.append(rf_train_accuracy)
    LSTM_RF_Test_Accuracy.append(rf_test_accuracy)
    LSTM_RF_F1_Accuracy.append(rf_f1_score)
    
    print(f"Epoch {epoch+1} - RF Results:")
    print(f"RF Train Accuracy: {rf_train_accuracy*100:.2f}%")
    print(f"RF Test Accuracy: {rf_test_accuracy*100:.2f}%")
    print(f"RF F1 Score: {rf_f1_score:.4f}")
    
    # 保存结果
    with open(r"./result/LSTM_RF/LSTM_RF_time_order.txt", "a") as f:
        f.write(f"{LSTM_RF_time_order[-1]:.3f},")
    with open(r"./result/LSTM_RF/LSTM_RF_Train_Accuracy.txt", "a") as f:
        f.write(f"{rf_train_accuracy:.6f},")
    with open(r"./result/LSTM_RF/LSTM_RF_Test_Accuracy.txt", "a") as f:
        f.write(f"{rf_test_accuracy:.6f},")
    with open(r"./result/LSTM_RF/LSTM_RF_F1_Accuracy.txt", "a") as f:
        f.write(f"{rf_f1_score:.6f},")


