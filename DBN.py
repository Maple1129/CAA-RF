import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 读取数据
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

# 转换为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

LSTM_time_order = [0.0]
# 定义DBN模型
class DBN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DBN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size

        # 构建多个隐藏层
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # 输出层
        self.output_layer = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = torch.relu(hidden_layer(x))
        return self.output_layer(x)


# 定义训练过程
def train(model, train_loader, criterion, optimizer, device):
    start = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    duration = time.time() - start
    LSTM_time_order.append(duration + LSTM_time_order[-1])
    with open(rf"./result/DBN/time_order.txt", "a") as f:
        f.write(f"{LSTM_time_order[-1]:.4f},")
    return running_loss / len(train_loader), correct / total


# 定义评估过程
def evaluate(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算测试准确率和F1分数
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1_score = f1_score(all_labels, all_preds, average='weighted')  # 使用加权平均计算F1分数

    return test_accuracy, test_f1_score


# 设置超参数
input_size = X_train.shape[1]  # 特征数
hidden_sizes = [128, 64, 32]  # 隐藏层大小
output_size = len(torch.unique(y_train_tensor))  # 类别数
learning_rate = 0.001
epochs = 2000

# 使用GPU加速训练（如果有的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = DBN(input_size, hidden_sizes, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 打开文件以便写入训练和测试准确率及F1分数
with open("Test_Accuracy_Train_Accuracy_F1.txt", "w") as f:
    # 训练和评估模型
    for epoch in range(epochs):
        with open(rf"./result/DBN/epoch_order.txt", "a") as f:
            f.write(f"{epoch+1},")
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_accuracy, test_f1_score = evaluate(model, test_loader, device)

        final_test_accuracy, final_test_f1_score = evaluate(model, test_loader, device)
        # 写入到文件

        with open(rf"./result/DBN/Train_Accuracy.txt", "a") as f:
            f.write(f"{train_accuracy:.4f},")
        with open(rf"./result/DBN/F1_accuracy.txt", "a") as f:
            f.write(f"{test_f1_score:.4f},")
        # 最终输出模型的测试准确率和F1分数
        with open(rf"./result/DBN/Test_Accuracy.txt", "a") as f:
            f.write(f"{test_accuracy:.4f},")


        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, "
              f"Test F1 Score: {test_f1_score:.4f}")


    print(f"Final Test Accuracy: {final_test_accuracy:.4f}, Final Test F1 Score: {final_test_f1_score:.4f}")
