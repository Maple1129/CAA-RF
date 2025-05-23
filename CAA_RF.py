import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler

sum_start_time = time.time()
class Data(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = self.features.iloc[index].values if isinstance(self.features, pd.DataFrame) else self.features[index]
        label = self.labels.iloc[index] if isinstance(self.labels, pd.Series) else self.labels[index]
        feature = feature.astype(np.float32)
        label = label.astype(np.int64)
        return torch.tensor(feature).unsqueeze(0), torch.tensor(label)


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, in_channels)
        query = self.query(x)  # (batch_size, sequence_length, in_channels)
        key = self.key(x)  # (batch_size, sequence_length, in_channels)
        value = self.value(x)  # (batch_size, sequence_length, in_channels)

        # Compute attention scores
        attn_scores = torch.bmm(query, key.transpose(1, 2))  # (batch_size, sequence_length, sequence_length)1
        attn_weights = self.softmax(attn_scores)  # (batch_size, sequence_length, sequence_length)
        # print(attn_scores)

        # Apply attention weights to the values
        attended_values = torch.bmm(attn_weights, value)  # (batch_size, sequence_length, in_channels)

        return attended_values


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        out_channels3 = 32

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1)

        # Add Dropout layers
        self.dropout = nn.Dropout(p=0.2)

        # 注意力层
        self.attention = SelfAttention(out_channels3 * 14)

        self.fc1 = nn.Linear(out_channels3 * 14, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)  # Apply dropout after first conv layer
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = x.unsqueeze(1)  # Add a sequence dimension for the attention layer
        x = self.attention(x)  # Apply attention
        x = x.squeeze(1)  # Remove the sequence dimension
        x = self.fc1(x)
        x = nn.PReLU()(x)

        x = self.fc2(x)
        return x


def train(model, train_data_loader, test_data_loader, criterion, optimizer, scheduler, epochs=1, ord=0, device=None):
    global epoch_order, CAARF_time_order, CAA_time_order, CAA_Train_Accuracy, CAA_Test_Accuracy, CAARF_Train_Accuracy, CAARF_Test_accuracy, F1_accuracy
    start_time = time.time()

    model.to(device)
    model.train()  # 设置模型为训练模式

    for epoch in range(epochs):
        correct_train = 0
        total_train = 0
        running_loss = 0.0

        # 训练阶段
        for features, labels in train_data_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        end_time = time.time()
        # 计算平均损失
        avg_loss = running_loss / len(train_data_loader)
        # 更新学习率调度器
        scheduler.step(avg_loss)
        end_time = time.time()
        print(f"Epoch {ord}, time:{end_time - start_time}")

        # 保存训练集的准确率
        epoch_order.append(ord)
        time_duration = end_time - start_time

        # # 输出训练集的信息
        # print(f"Epoch {ord}, Train Loss: {avg_loss},Train Accuracy: {100 * correct_train / total_train}%")

        # 测试阶段
        model.eval()  # 设置模型为评估模式
        correct_test = 0
        total_test = 0
        all_predicted = []
        all_labels = []
        with torch.no_grad():  # 不计算梯度
            for features, labels in test_data_loader:
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_predicted, average='weighted')

        accuracy_train = correct_train / total_train
        accuracy_test = correct_test / total_test
        # 保存测试集的准确率
        CAA_F1_Accuracy.append(f1)
        CAA_time_order.append(time_duration + CAA_time_order[-1])
        CAA_Test_Accuracy.append(accuracy_test)
        CAA_Train_Accuracy.append(accuracy_train)

        # 输出训练集的信息
        print(f"         Train Accuracy: {100 * accuracy_train :.2f}, Train Loss: {avg_loss}")
        # 输出测试集的信息
        print(f"         Test  Accuracy: {100 * accuracy_test:.2f}")

        # with open(r"./result/CAA/0922F1_accuracy_time_epocu.txt", "a") as f:
        #     f.write(f"------Epoch {ord}\nepoch_order:{epoch_order}\ntime_order:{time_order}\nTrain_Accuracy:{Train_Accuracy}\nTest_Accuracy:{Test_Accuracy}")
        with open(r"./result/CAA/epoch_order.txt", "a") as f:
            f.write(f"{epoch_order[-1]},")
        with open(r"./result/CAA/CAA_time_order.txt", "a") as f:
            f.write(f"{CAA_time_order[-1]:.6f},")
        with open(r"./result/CAA/CAA_Train_Accuracy.txt", "a") as f:
            f.write(f"{CAA_Train_Accuracy[-1]:.6f},")
        with open(r"./result/CAA/CAA_Test_Accuracy.txt", "a") as f:
            f.write(f"{CAA_Test_Accuracy[-1]:.6f},")
        with open(r"./result/CAA/CAA_F1_Accuracy.txt", "a") as f:
            f.write(f"{CAA_F1_Accuracy[-1]:.6f},")


def remove_last_layer(model):
    model.fc2 = nn.Identity()  # 将最后一层替换为Identity层，使得输出不变
    return model


# 恢复模型的fc2层
def restore_last_layer(model, original_layer):
    model.fc2 = original_layer
    return model


# 提取特征
def extract_features(model, data_loader):
    model.eval()  # 设置模型为评估模式
    all_features = []
    all_labels = []

    with torch.no_grad():  # 不计算梯度
        for features, labels in data_loader:
            outputs = model(features)
            all_features.append(outputs.cpu().numpy())  # 将输出转换为 NumPy 数组
            all_labels.append(labels.cpu().numpy())

    # 将所有特征和标签合并成一个 NumPy 数组
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


global epoch_order, CAARF_time_order, CAA_time_order, CAA_Train_Accuracy, CAA_Test_Accuracy, CAA_F1_Accuracy, CAARF_Train_Accuracy, CAARF_Test_Accuracy, CAARF_F1_Accuracy
epoch_order = []
CAARF_time_order = [0.0]
CAA_time_order = [0.0]
CAA_Train_Accuracy = []
CAA_Test_Accuracy = []
CAARF_Train_Accuracy = []
CAARF_Test_Accuracy = []
CAARF_F1_Accuracy = []
CAA_F1_Accuracy = []
# 加载指定数据集
start = time.time()
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

# 创建数据集
dataset_train = Data(X_train, y_train)
dataset_test = Data(X_test, y_test)

data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32, shuffle=True)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

t = 1

train(model, data_loader, test_data_loader, criterion, optimizer, scheduler, epochs=t, ord=1)

# 保存原始的fc2层
original_fc2 = model.fc2
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 前面已经训练过了一次
for ep in range(20000):
    # 去掉最后一层
    feature_extractor = remove_last_layer(model)
    # 从神经网络中提取特征
    features, labels = extract_features(feature_extractor, data_loader)  # 使用训练数据提取特征
    start = time.time()

    rf.fit(features, labels)
    duration = time.time() - start

    print(f'         time:{duration:.4f}')

    features_test, labels = extract_features(feature_extractor, data_loader)  # 使用测试数据提取特征
    y_pred = rf.predict(features_test)  # 使用测试特征来进行预测
    accuracy_train = accuracy_score(labels, y_pred)
    print(f'         train accuracy: {accuracy_train:.6f}')

    # 验证模型
    features_test, labels_test = extract_features(feature_extractor, test_data_loader)  # 使用测试数据提取特征
    y_pred = rf.predict(features_test)  # 使用测试特征来进行预测
    accuracy_test = accuracy_score(labels_test, y_pred)

    # 计算F1分数
    f1_macro = f1_score(labels_test, y_pred, average='macro')
    # Macro-average F1 Score：它分别计算每个类别的F1分数，然后取平均值。这种方法对于类别数目较多且希望每个类别都有相同重要性的场景很有用。

    CAARF_time_order.append(CAA_time_order[-1] + duration)
    CAARF_Test_Accuracy.append(accuracy_test)
    CAARF_Train_Accuracy.append(accuracy_train)
    CAARF_F1_Accuracy.append(f1_macro)
    print(f'         F1          Score: {f1_macro:.6f}')
    print(f"         RF    Accuracy: {accuracy_test * 100:.2f}%")

    with open(r"./result/CAA/CAARF_time_order.txt", "a") as f:
        f.write(f"{CAARF_time_order[-1]:.3f},")
    with open(r"./result/CAA/CAARF_F1_Accuracy.txt", "a") as f:
        f.write(f"{CAARF_F1_Accuracy[-1]:.6f},")
    with open(r"./result/CAA/CAARF_Test_Accuracy.txt", "a") as f:
        f.write(f"{CAARF_Test_Accuracy[-1]:.6f},")
    with open(r"./result/CAA/CAARF_Train_Accuracy.txt", "a") as f:
        f.write(f"{CAARF_Train_Accuracy[-1]:.6f},")

    # 恢复模型的fc2层
    model = restore_last_layer(model, original_fc2)

    print(f"程序运行持续时间:{(time.time() - sum_start_time) / 60.0:.4f}分")
    # 继续使用恢复后的模型进行训练或测试
    train(model, data_loader, test_data_loader, criterion, optimizer=optimizer, scheduler=scheduler, epochs=1,
          ord=ep + 2)
