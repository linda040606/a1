import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, TimeDistributed, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 数据加载与维度修正
def load_data(mat_path):
    mat = sio.loadmat(mat_path)
    x_train = mat['x_train']  # 原始形状 (时间点, 通道, 试验)
    y_train = mat['y_train'].ravel()
    x_test = mat['x_test']
    
    # 转置为 (试验数, 时间步, 通道数) 并添加维度
    x_train = np.transpose(x_train, (2, 0, 1))  # (316, 50, 28)
    x_test = np.transpose(x_test, (2, 0, 1))    # (100, 50, 28)
    
    # 重塑为Conv2D输入格式 (时间步, 高度=1, 宽度=28, 通道=1)
    x_train = x_train.reshape(*x_train.shape, 1)  # (316, 50, 28, 1)
    x_test = x_test.reshape(*x_test.shape, 1)     # (100, 50, 28, 1)
    return x_train, y_train, x_test

# 2. 数据预处理
def preprocess(x_train, x_test):
    # 标准化每个通道
    scaler = StandardScaler()
    n_samples, n_timesteps, n_channels, _ = x_train.shape
    x_train = scaler.fit_transform(x_train.reshape(-1, n_channels)).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, n_channels)).reshape(x_test.shape)
    return x_train, x_test

# 3. 数据增强（修正版）
def augment_data(x, y, noise_factor=0.05, max_shift=2):
    augmented_x, augmented_y = [], []
    for xi, yi in zip(x, y):
        # 原始数据 (50, 28, 1)
        augmented_x.append(xi)
        augmented_y.append(yi)
        
        # 添加噪声
        noisy = xi + np.random.normal(0, noise_factor, xi.shape)
        augmented_x.append(noisy)
        augmented_y.append(yi)
        
        # 时间偏移（仅在时间维度操作）
        shift = np.random.randint(-max_shift, max_shift+1)
        if shift == 0:
            continue
            
        if shift > 0:
            # 正向偏移：填充前部并截取后部
            padded = np.pad(xi, ((shift,0), (0,0), (0,0)), mode='edge')  # 时间维度填充
            shifted = padded[shift:, :, :]
        else:
            # 负向偏移：填充后部并截取前部
            abs_shift = abs(shift)
            padded = np.pad(xi, ((0,abs_shift), (0,0), (0,0)), mode='edge')
            shifted = padded[:-abs_shift, :, :]
            
        augmented_x.append(shifted)
        augmented_y.append(yi)
    return np.array(augmented_x), np.array(augmented_y)

# 4. 构建适配电极拓扑的混合模型
def build_model(input_shape):
    inputs = Input(shape=input_shape)  # input_shape = (50, 28, 1)
    
    # 增加电极空间维度（将通道28映射为1x28的2D空间）
    x = Reshape((50, 1, 28, 1))(inputs)  # 输出形状 (None, 50, 1, 28, 1)
    
    # CNN部分：提取空间特征
    x = TimeDistributed(Conv2D(16, kernel_size=(1,3), activation='relu'))(x)  # 1x3卷积核扫描电极宽度
    x = TimeDistributed(MaxPooling2D(pool_size=(1,2)))(x)  # 宽度减半 → (None, 50, 1, 14, 16)
    x = TimeDistributed(Dropout(0.3))(x)
    
    x = TimeDistributed(Conv2D(32, kernel_size=(1,3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(1,2)))(x)  # → (None, 50, 1, 7, 32)
    x = TimeDistributed(Flatten())(x)  # 展平 → (None, 50, 1*7*32=224)
    
    # RNN部分：建模时间依赖
    x = LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = LSTM(32, return_sequences=False)(x)
    
    # 分类层
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 主流程
if __name__ == "__main__":
    # 参数
    mat_path = 'data.mat'
    batch_size = 16
    epochs = 50
    
    # 加载数据
    x_train, y_train, x_test = load_data(mat_path)
    print("原始数据形状:", x_train.shape)  # 应为 (316, 50, 28, 1)
    
    # 预处理
    x_train, x_test = preprocess(x_train, x_test)
    
    # 数据增强
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    print("增强后训练集形状:", x_train_aug.shape)
    
    # 划分验证集
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train_aug, y_train_aug, test_size=0.2, stratify=y_train_aug
    )
    
    # 构建模型
    model = build_model(input_shape=(50, 28, 1))
    model.summary()
    
    # 训练
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        x_train_split, y_train_split,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # 预测测试集
    y_pred = (model.predict(x_test) > 0.5).astype(int)
    print("测试集预测结果:", y_pred.flatten())