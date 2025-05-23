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
from sklearn.ensemble import RandomForestClassifier
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


# 定义一个函数来提取VAE的中间层特征
def extract_vae_features(model, data_loader):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            _, mu, _ = model(features)  # 提取中间层特征
            all_features.append(mu.cpu().numpy())  # 使用均值作为特征
            all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels



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
    
    global opech_order, LSTM_time_order, Train_Accuracy, Test_Accuracy, F1_Accuracy
    opech_order.append(ord)
    time_order.append(duration + time_order[-1])

    # 提取VAE中间层特征
    vae_features_train, vae_labels_train = extract_vae_features(vae, train_loader)
    # 使用随机森林对VAE中间层特征进行分类
    rf_vae.fit(vae_features_train, vae_labels_train)
    # 评估随机森林在VAE中间层特征上的表现
    vae_y_pred_train = rf_vae.predict(vae_features_train)
    vae_accuracy_train = accuracy_score(vae_labels_train, vae_y_pred_train)
    vae_f1_train = f1_score(vae_labels_train, vae_y_pred_train, average='weighted')

    with open(r"./result/Variational_Auto_RF/Train_Accuracy.txt", "a") as f:
        f.write(f"{vae_accuracy_train:.6f},")
    with open(r"./result/Variational_Auto_RF/epoch_order.txt", "a") as f:
        f.write(f"{opech_order[-1]},")
    with open(r"./result/Variational_Auto_RF/time_order.txt", "a") as f:
        f.write(f"{time_order[-1]:.6f},")

    print(f'Epoch {ord}, Loss: {train_loss / len(train_loader.dataset)}')
    print(f'VAE中间层特征 + RF 训练集准确率: {vae_accuracy_train:.2f}, F1得分: {vae_f1_train:.2f}')




    

def test(vae):
    # 提取VAE的编码特征
    vae.eval()
    # 提取VAE中间层特征
    vae_features_train, vae_labels_train = extract_vae_features(vae, train_loader)
    vae_features_test, vae_labels_test = extract_vae_features(vae, DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size))

    # 使用随机森林对VAE中间层特征进行分类
    rf_vae.fit(vae_features_train, vae_labels_train)

    # 评估随机森林在VAE中间层特征上的表现
    vae_y_pred_test = rf_vae.predict(vae_features_test)

    vae_accuracy_test = accuracy_score(vae_labels_test, vae_y_pred_test)
    vae_f1_test = f1_score(vae_labels_test, vae_y_pred_test, average='weighted')

    Test_Accuracy.append(vae_accuracy_test)
    F1_Accuracy.append(vae_f1_test)

    with open(r"./result/Variational_Auto_RF/Test_Accuracy.txt", "a") as f:
        f.write(f"{vae_accuracy_test:.6f},")
    with open(r"./result/Variational_Auto_RF/F1_Accuracy.txt", "a") as f:
        f.write(f"{vae_f1_test:.6f},")
    print(f'VAE + RF 测试集准确率: {vae_accuracy_test:.2f}, F1得分: {vae_f1_test:.2f}')



# 初始化全局变量
opech_order = []
time_order = [0.0]
Train_Accuracy = []
Test_Accuracy = []
F1_Accuracy = []

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
    rf_vae = RandomForestClassifier(n_estimators=100, random_state=42)
    train(vae, train_loader, optimizer, epoch+1)
    # 训练VAE
    test(vae)





