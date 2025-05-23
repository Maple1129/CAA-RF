import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import ADASYN

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

# 将数据转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 数据加载器
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder_fc1 = nn.Linear(input_dim, 64)
        self.encoder_fc2_mean = nn.Linear(64, latent_dim)
        self.encoder_fc2_log_var = nn.Linear(64, latent_dim)

        # 解码器
        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, input_dim)

    def encode(self, x):
        h = torch.relu(self.encoder_fc1(x))
        z_mean = self.encoder_fc2_mean(h)
        z_log_var = self.encoder_fc2_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z):
        h = torch.relu(self.decoder_fc1(z))
        x_decoded = torch.sigmoid(self.decoder_fc2(h))
        return x_decoded

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z)
        return x_decoded, z_mean, z_log_var

# 定义VAE的损失函数
def vae_loss_function(x, x_decoded, z_mean, z_log_var):
    reconstruction_loss = nn.functional.mse_loss(x_decoded, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return reconstruction_loss + kl_loss

def train(vae, train_loader, optimizer, ord):
    # 训练VAE
    vae.train()
    train_loss = 0
    start = time.time()
    for batch_x, _ in train_loader:
        optimizer.zero_grad()
        x_decoded, z_mean, z_log_var = vae(batch_x)
        loss = vae_loss_function(batch_x, x_decoded, z_mean, z_log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    duration = time.time() - start
    vae.eval()
    global opech_order, LSTM_time_order, Train_Accuracy, Test_Accuracy, F1_accuracy
    opech_order.append(ord)
    time_order.append(duration + time_order[-1])

    with open(r"./result/Variational_Auto/epoch_order.txt", "a") as f:
        f.write(f"{opech_order[-1]},")
    with open(r"./result/Variational_Auto/time_order.txt", "a") as f:
        f.write(f"{time_order[-1]:.6f},")

    print(f'Epoch {ord}, Loss: {train_loss / len(train_loader.dataset)}')

def test(vae):
    # 提取VAE的编码特征
    vae.eval()
    with torch.no_grad():
        X_train_encoded, _ = vae.encode(X_train_tensor)
        X_test_encoded, _ = vae.encode(X_test_tensor)

    # 转换为numpy数组，方便后续使用sklearn
    X_train_encoded = X_train_encoded.numpy()
    X_test_encoded = X_test_encoded.numpy()

    # 使用逻辑回归进行分类
    classifier = LogisticRegression()
    classifier.fit(X_train_encoded, y_train)
    y_train_pred = classifier.predict(X_train_encoded)
    y_test_pred = classifier.predict(X_test_encoded)

    # 计算准确率
    accuracy_test = accuracy_score(y_test, y_test_pred)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_f1 = f1_score(y_test, y_test_pred, average='macro')

    Test_Accuracy.append(accuracy_test)
    Train_Accuracy.append(accuracy_train)
    F1_accuracy.append(accuracy_f1)

    with open(r"./result/Variational_Auto/Test_Accuracy.txt", "a") as f:
        f.write(f"{accuracy_test:.6f},")
    with open(r"./result/Variational_Auto/Train_Accuracy.txt", "a") as f:
        f.write(f"{accuracy_train:.6f},")
    with open(r"./result/Variational_Auto/F1_accuracy.txt", "a") as f:
        f.write(f"{accuracy_f1:.6f},")

    print("测试集分类准确率: {:.6f}%".format(accuracy_test * 100))

# 初始化全局变量
opech_order = []
time_order = [0.0]
Train_Accuracy = []
Test_Accuracy = []
F1_accuracy = []

# 设置参数
input_dim = X_train.shape[1]  # 使用处理后的特征维度
latent_dim = 2  # 潜在空间的维度
vae = VAE(input_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=0.001)
epochs = 7000
# patience = 100
best_loss = float('inf')
counter = 0

for epoch in range(epochs):
    train(vae, train_loader, optimizer, epoch+1)
    # 训练VAE
    test(vae)
