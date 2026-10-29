import numpy as np

class DenseLayer:
    """Fully connected linear layer with He (kaiming) initialization."""
    def __init__(self, in_features: int, out_features: int):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))

        self.x = None
        self.dw = None
        self.db = None

        self.m_w, self.v_W = np.zeros_like(self.W), np.zeros_like(self.W)
        self.m_b, self.v_b = np.zeros_like(self.b), np.zeros_like(self.b)

    def forward(self, x : np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        self.dw = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis=0, keepdims=True)
        return np.dot(d_out, self.W.T)

class ReLU:
    """Rectified Linear Unit activation function."""

    def __init__(self):
        self.z = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.z = z
        return np.maximum(0, z)

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        return d_out * (self.z > 0).astype(np.float64)

class SoftmaxCrossEntropyLoss:
    """Fused Softmax activation and Categorial Cross-Entropy loss.

    Fusing eliminates numerical instability and simplifies the backward gradient
    to: (y_hat - y) / batch_size
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        self.y_true = y_true
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        self.y_pred = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        batch_size = logits.shape[0]

        eps = 1e-15
        probs_clamped = np.clip(self.y_pred, eps, 1.0 - eps)

        loss = -np.sum(y_true * np.log(probs_clamped)) / batch_size
        return float(loss)

    def backward(self) -> np.ndarray:
        batch_size = self.y_true.shape[0]

        return (self.y_pred - self.y_true) / batch_size

class AdamOptimizer:
    """Adam optimizer tracking first and second uncentered moment estimates."""

    def __init__(
            self,
            layers: list,
            lr: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
    ):
        self.layers = [layer for layer in layers if isinstance(layer, DenseLayer)]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self):
        self.t += 1
        for layer in self.layers:
            layer.m_w = (
                self.beta1 * layer.m_w + (1.0 - self.beta1) * layer.dw
            )
            layer.v_W = (
                self.beta2 * layer.v_W + (1.0 - self.beta2) * (layer.dw**2)
            )

            m_W_hat = layer.m_w / (1.0 - self.beta1**self.t)
            v_W_hat = layer.v_W / (1.0 - self.beta2**self.t)
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)

            layer.m_b = (
                self.beta1 * layer.m_b + (1.0 - self.beta1) * layer.db
            )
            layer.v_b = (
                self.beta2 * layer.v_b + (1.0 - self.beta2) * (layer.db**2)
            )
            m_b_hat = layer.m_b / (1.0 - self.beta1**self.t)
            v_b_hat = layer.v_b / (1.0 - self.beta2**self.t)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


class NeuralNetwork:
    """Sequential model coordinating forward propagation and backpropagation."""
    def __init__(self, layer_list: list):
        self.layers = layer_list
        self.loss_fn = SoftmaxCrossEntropyLoss()

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad: np.ndarray) -> np.ndarray:
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 150,
            batch_size: int = 32,
            lr: float = 0.01,
    ):
        optimizer = AdamOptimizer(self.layers, lr=lr)
        n_samples = X.shape[0]

        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = int(np.ceil(n_samples / batch_size))

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, n_samples)
                X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]

                logits = self.forward(X_batch)
                loss = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss * (end - start)

                d_logits = self.loss_fn.backward()
                self.backward(d_logits)

                optimizer.step()
            total_loss = epoch_loss / n_samples
            if epoch % 30 == 0 or epoch == 1:
                preds = np.argmax(self.forward(X), axis=1)
                targets = np.argmax(y, axis=1)
                accuracy = np.mean(preds == targets) * 100
                print(f"Epoch {epoch:03d}/{epochs:03d} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    np.random.seed(42)

    points_per_class = 150
    dimensions = 2
    classes = 3

    X = np.zeros((points_per_class * classes, dimensions))
    y = np.zeros((points_per_class * classes, classes))

    for c in range(classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1.0, points_per_class)
        t = (np.linspace(c * 4.0, (c+1) * 4.0, points_per_class) + np.random.randn(points_per_class) ** 0.2)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix, c] = 1.0


    mlp = NeuralNetwork(
        [
            DenseLayer(in_features=2, out_features=64),
            ReLU(),
            DenseLayer(in_features=64, out_features=32),
            ReLU(),
            DenseLayer(in_features=32, out_features=3),
        ]
    )
    print("Training 3-Layer MLP on non-linear spiral data...")
    mlp.fit(X, y, epochs=150, batch_size=32, lr=0.01)