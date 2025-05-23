import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
import pandas as pd
import numpy as np

# 定义评价函数
'''数据加载不同'''

def evaluate_classifier(classifier, data_loader):
    '''计算准确率，F1'''
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # 计算准确率和F1分数
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    return accuracy, f1_macro

# 数据集类定义
class Data(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = self.features[index].astype(np.float32)
        label = self.labels[index].astype(np.int64)
        return torch.tensor(feature), torch.tensor(label)

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

# 读取数据文件路径
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

# 创建数据集对象
dataset_train = Data(X_train, y_train)
dataset_test = Data(X_test, y_test)

# 创建数据加载器
train_data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)

# 计算类别权重以应对类别不平衡
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# 初始化生成器、判别器、分类器
noise_dim = 10  # 噪声的维度
learning_rate = 0.0002
epochs = 5000

opech_order = []
LSTM_time_order = [0.0]
Train_Accuracy = []
LSTM_Test_Accuracy = []
f1_accuracy = []

generator = Generator(noise_dim, X_train.shape[1])
discriminator = Discriminator(X_train.shape[1])
classifier = Classifier(X_train.shape[1])

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类损失用于判别器
classifier_criterion = nn.CrossEntropyLoss(weight=class_weights)  # 加入类别权重的多分类损失
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_C = optim.Adam(classifier.parameters(), lr=1e-4)  # 分类器使用较小的学习率

# 训练过程
for epoch in range(epochs):

    start_time = time.time()
    correct_train = 0
    total_train = 0

    # 使用数据加载器迭代数据
    for batch_idx, (real_samples, y_train_batch) in enumerate(train_data_loader):

        batch_size = real_samples.size(0)
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 真实数据
        real_output = discriminator(real_samples)
        d_real_loss = criterion(real_output, real_labels)

        # 生成的假数据
        noise = torch.randn(batch_size, noise_dim)
        fake_samples = generator(noise)
        fake_output = discriminator(fake_samples.detach())
        d_fake_loss = criterion(fake_output, fake_labels)

        # 判别器总损失
        d_loss = d_real_loss + d_fake_loss
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()

        fake_output = discriminator(fake_samples)
        g_loss = criterion(fake_output, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # 训练分类器
        optimizer_C.zero_grad()

        classifier_output = classifier(real_samples)
        c_loss = classifier_criterion(classifier_output, y_train_batch)
        c_loss.backward()
        optimizer_C.step()

        # 计算准确率
        _, predicted = torch.max(classifier_output, 1)
        correct_train += (predicted == y_train_batch).sum().item()

    end_time = time.time()

    # 使用测试集计算准确率
    test_accuracy, f1_macro = evaluate_classifier(classifier, test_data_loader)

    duration = end_time - start_time
    accuracy_train = correct_train / len(dataset_train)
    Train_Accuracy.append(accuracy_train)
    LSTM_time_order.append(LSTM_time_order[-1] + duration)
    opech_order.append(epoch + 1)

    LSTM_Test_Accuracy.append(test_accuracy)
    f1_accuracy.append(f1_macro)

    print(f'Epoch [{epoch}/{epochs}] | Test Accuracy: {test_accuracy:.6f} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()} | C Loss: {c_loss.item()}')

    # 保存数据的部分保持不变...
    with open(r"./result/GANs/opech_order.txt", "a") as f:
        f.write(f"{epoch+1},")
    with open(r"./result/GANs/time_order.txt", "a") as f:
        f.write(f"{LSTM_time_order[-1]:.6f},")
    with open(r"./result/GANs/Train_Accuracy.txt", "a") as f:
        f.write(f"{accuracy_train:.6f},")
    with open(r"./result/GANs/Test_Accuracy.txt", "a") as f:
        f.write(f"{LSTM_Test_Accuracy[-1]:.6f},")
    with open(r"./result/GANs/F1_accuracy.txt", "a") as f:
        f.write(f"{f1_accuracy[-1]:.6f},")

print("Training completed.")