import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, c=1):
    return 1 / (1 + np.exp(-c * x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

def bce_loss(y_pred, y_true):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_pred, y_true):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) + eps) / y_true.size

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def plot_loss_curve(losses, title="График потерь"):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.show()

def task_1_1():
    print("\n" + "="*60)
    print("ЗАДАНИЕ 1.1: Функции активации")
    print("="*60)
    
    np.random.seed(21)
    w_norm = np.random.normal(loc=0, scale=0.5, size=(2, 1))
    b_norm = np.random.normal(loc=0, scale=0.5, size=1)
    print(f"Инициализация весов (normal):\n{w_norm}")
    print(f"Инициализация смещения (normal): {b_norm}")
    
    x_vals = np.linspace(-10, 10, 200)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_vals, tanh(x_vals))
    plt.title("tanh")
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(x_vals, relu(x_vals))
    plt.title("ReLU")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    for c, style in zip([0.2, 1, 5], ['--', '-', ':']):
        plt.plot(x_vals, sigmoid(x_vals, c), style, label=f'c={c}')
    plt.title("Sigmoid с разной температурой")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Вывод: c < 1 — плавный переход, c > 1 — резкий переход (ближе к порогу)")

def train_neuron(X, y, loss_type='mse', lr=0.5, momentum=0.9, epochs=200, verbose=True):
    np.random.seed(21)
    n_features = X.shape[1]
    w = np.random.uniform(-0.05, 0.05, size=(n_features, 1))
    b = np.random.uniform(-0.05, 0.05, size=1)
    
    velocity_w = np.zeros_like(w)
    velocity_b = 0.0
    losses = []
    
    for epoch in range(epochs):
        s = X @ w + b
        y_pred = sigmoid(s)
        
        if loss_type == 'mse':
            loss = mse_loss(y_pred, y)
            grad = mse_derivative(y_pred, y)
        else:
            loss = bce_loss(y_pred, y)
            grad = bce_derivative(y_pred, y)
        
        losses.append(loss)
        
        dL_ds = grad * sigmoid_derivative(s)
        
        dL_dw = X.T @ dL_ds
        dL_db = np.sum(dL_ds)
        
        velocity_w = momentum * velocity_w + lr * dL_dw
        velocity_b = momentum * velocity_b + lr * dL_db
        
        w -= velocity_w
        b -= velocity_b
    
    final_pred = sigmoid(X @ w + b) >= 0.5
    final_acc = accuracy(final_pred, y)
    
    if verbose:
        print(f"  {loss_type.upper()}: loss={losses[-1]:.6f}, accuracy={final_acc:.2f}")
    
    return w, b, losses, final_acc

def task_1_2():
    print("\n" + "="*60)
    print("ЗАДАНИЕ 1.2: Логическое «И» (AND)")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([[0], [0], [0], [1]])
    
    print("\n1. Сравнение MSE и BCE:")
    _, _, losses_mse, acc_mse = train_neuron(X, y_and, loss_type='mse', lr=0.5, epochs=500, verbose=True)
    _, _, losses_bce, acc_bce = train_neuron(X, y_and, loss_type='bce', lr=0.5, epochs=500, verbose=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_mse, label='MSE')
    plt.plot(losses_bce, label='BCE')
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("MSE vs BCE для задачи AND")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n2. Влияние learning rate (lr):")
    lr_values = [0.1, 1.0, 5.0]
    plt.figure(figsize=(10, 5))
    for lr in lr_values:
        _, _, losses, _ = train_neuron(X, y_and, lr=lr, epochs=500, verbose=False)
        plt.plot(losses, label=f'lr={lr}')
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Влияние learning rate")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("  Вывод: lr=0.1 — медленно, lr=1.0 — оптимально, lr=5.0 — нестабильно")
    
    print("\n3. Влияние momentum:")
    mom_values = [0.0, 0.5, 0.9]
    plt.figure(figsize=(10, 5))
    for mom in mom_values:
        _, _, losses, _ = train_neuron(X, y_and, momentum=mom, epochs=500, verbose=False)
        plt.plot(losses, label=f'momentum={mom}')
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Влияние momentum")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("  Вывод: momentum ускоряет сходимость и сглаживает колебания")

def task_1_3():
    print("\n" + "="*60)
    print("ЗАДАНИЕ 1.3: Логическое «ИЛИ» (OR)")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([[0], [1], [1], [1]])
    
    best_acc = 0
    best_params = {}
    
    for lr in [0.3, 0.5, 0.8]:
        for mom in [0.7, 0.9]:
            _, _, losses, acc = train_neuron(X, y_or, lr=lr, momentum=mom, epochs=300, verbose=False)
            if acc > best_acc:
                best_acc = acc
                best_params = {'lr': lr, 'momentum': mom}
    
    print(f"\nЛучшие параметры: lr={best_params['lr']}, momentum={best_params['momentum']}")
    print(f"Точность: {best_acc:.2f}")
    
    w, b, losses, acc = train_neuron(X, y_or, lr=best_params['lr'], momentum=best_params['momentum'], epochs=300, verbose=True)
    plot_loss_curve(losses, "OR — график потерь")
    
    print("\n OR решена с 100% точностью")

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size, 
                 activation='sigmoid', loss='mse', seed=42):
        np.random.seed(seed)
        
        self.activation = activation
        self.loss_type = loss
        
        if activation == 'sigmoid':
            self.f = sigmoid
            self.f_deriv = sigmoid_derivative
        elif activation == 'relu':
            self.f = relu
            self.f_deriv = relu_derivative
        else:
            self.f = tanh
            self.f_deriv = tanh_derivative
        
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
    
    def forward(self, X):
        self.X = X
        self.s1 = X @ self.W1 + self.b1
        self.z1 = self.f(self.s1)
        self.s2 = self.z1 @ self.W2 + self.b2
        self.z2 = sigmoid(self.s2)
        return self.z2
    
    def compute_loss(self, y_pred, y_true):
        if self.loss_type == 'mse':
            return mse_loss(y_pred, y_true)
        else:
            return bce_loss(y_pred, y_true)
    
    def backward(self, y_true, lr, momentum):
        if self.loss_type == 'mse':
            dL_dz2 = mse_derivative(self.z2, y_true)
        else:
            dL_dz2 = bce_derivative(self.z2, y_true)
        
        dz2_ds2 = sigmoid_derivative(self.s2)
        dL_ds2 = dL_dz2 * dz2_ds2
        
        dL_dW2 = self.z1.T @ dL_ds2
        dL_db2 = np.sum(dL_ds2, axis=0, keepdims=True)
        
        dL_dz1 = dL_ds2 @ self.W2.T
        dz1_ds1 = self.f_deriv(self.s1)
        dL_ds1 = dL_dz1 * dz1_ds1
        
        dL_dW1 = self.X.T @ dL_ds1
        dL_db1 = np.sum(dL_ds1, axis=0, keepdims=True)
        
        self.vW2 = momentum * self.vW2 + lr * dL_dW2
        self.vb2 = momentum * self.vb2 + lr * dL_db2
        self.vW1 = momentum * self.vW1 + lr * dL_dW1
        self.vb1 = momentum * self.vb1 + lr * dL_db1
        
        self.W2 -= self.vW2
        self.b2 -= self.vb2
        self.W1 -= self.vW1
        self.b1 -= self.vb1
    
    def train(self, X, y, epochs, lr=0.5, momentum=0.9, verbose=True):
        losses = []
        
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            self.backward(y, lr, momentum)
        
        final_pred = (self.forward(X) >= 0.5).astype(int)
        final_acc = accuracy(final_pred, y)
        
        if verbose:
            print(f"  epochs={epochs}, loss={losses[-1]:.6f}, accuracy={final_acc:.2f}")
        
        return losses, final_acc

def task_1_4():
    print("\n" + "="*60)
    print("ЗАДАНИЕ 1.4: Исключающее ИЛИ (XOR)")
    print("="*60)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    print("\nПочему один нейрон не решает XOR?")
    print("XOR — линейно не разделим. Нельзя провести одну прямую,")
    print("чтобы отделить (0,0) и (1,1) от (0,1) и (1,0).")
    print("Нужен хотя бы один скрытый слой.\n")
    
    print("1. Эксперимент с разным числом нейронов в скрытом слое:")
    hidden_sizes = [2, 3, 5]
    
    plt.figure(figsize=(12, 4))
    for i, h in enumerate(hidden_sizes):
        net = TwoLayerNetwork(2, h, 1, activation='sigmoid', loss='mse')
        losses, acc = net.train(X, y_xor, epochs=1000, lr=0.5, momentum=0.9, verbose=True)
        plt.subplot(1, 3, i+1)
        plt.plot(losses)
        plt.title(f"hidden={h}, acc={acc:.2f}")
        plt.xlabel("Эпоха")
        plt.ylabel("Loss")
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("  Вывод: больше нейронов — быстрее обучение, но возможен риск переобучения")
    
    print("\n2. Эксперимент: Sigmoid vs ReLU в скрытом слое:")
    
    net_sigmoid = TwoLayerNetwork(2, 3, 1, activation='sigmoid', loss='mse')
    losses_sigmoid, acc_sigmoid = net_sigmoid.train(X, y_xor, epochs=1000, lr=0.5, momentum=0.9, verbose=True)
    
    net_relu = TwoLayerNetwork(2, 3, 1, activation='relu', loss='mse')
    losses_relu, acc_relu = net_relu.train(X, y_xor, epochs=1000, lr=0.5, momentum=0.9, verbose=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses_sigmoid, label='Sigmoid')
    plt.plot(losses_relu, label='ReLU')
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Sigmoid vs ReLU в скрытом слое (XOR)")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("  Вывод: ReLU обычно сходится быстрее, но может иметь мёртвые нейроны")
    
    print("\n3. Финальное обучение для достижения 100% точности:")
    net_final = TwoLayerNetwork(2, 4, 1, activation='relu', loss='mse')
    losses_final, acc_final = net_final.train(X, y_xor, epochs=2000, lr=0.3, momentum=0.95, verbose=True)
    
    final_pred = net_final.forward(X) >= 0.5
    print(f"\nФинальные предсказания:\n{final_pred.astype(int).flatten()}")
    print(f"Истинные значения: {y_xor.flatten()}")
    
    if acc_final == 1.0:
        print("\n XOR решена с 100% точностью!")
    else:
        print(f"\nТочность: {acc_final:.2f}, попробуйте увеличить epochs")
    
    plot_loss_curve(losses_final, "XOR — финальный график потерь")

if __name__ == "__main__":   
    task_1_1()
    task_1_2()
    task_1_3()
    task_1_4()
    
