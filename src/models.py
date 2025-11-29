import numpy as np
from tqdm import trange

# ==============================
# Các HÀM CƠ BẢN CHO MÔ HÌNH & METRIC
def sigmoid(z):
    """
    Hàm sigmoid ổn định số học:
    - Clip z để tránh overflow trong exp(-z)
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_prob):
    """
    Binary cross-entropy cho bài toán nhị phân.
    - Clip y_prob để tránh log(0)
    """
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def roc_auc_score(y_true, y_prob):
    """
    Tự cài AUC (ROC-AUC) theo U-statistic:
    - Sort theo y_prob
    - Tính tổng rank của positive
    - Chuyển về AUC
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    ranks = np.arange(1, len(y_true_sorted) + 1)
    rank_sum_pos = np.sum(ranks[y_true_sorted == 1])

    U = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    auc = U / (n_pos * n_neg)
    return float(auc)


def load_csv_as_str(path, delimiter=",", has_header=True, encoding="utf-8"):
    """
    Đọc CSV đơn giản:
    - Trả về header (list) và data (np.ndarray[str])
    """
    with open(path, "r", encoding=encoding) as f:
        lines = f.read().strip().split("\n")

    if has_header:
        header = lines[0].split(delimiter)
        rows = lines[1:]
    else:
        header = None
        rows = lines

    data = [row.split(delimiter) for row in rows]
    return header, np.array(data, dtype=str)


def load_xy_from_clean_csv(csv_path, target_col="target"):
    """
    Đọc file CSV đã clean:
    - target_col là cột nhãn
    - X, y đều là float
    - kèm theo danh sách tên feature
    """
    header, data_str = load_csv_as_str(csv_path)
    if target_col not in header:
        raise ValueError(f"Không thấy cột target '{target_col}'.")
    t_idx = header.index(target_col)

    y_raw = data_str[:, t_idx]
    y = y_raw.astype(float)

    feat_idx = [i for i in range(len(header)) if i != t_idx]
    X_raw = data_str[:, feat_idx]
    X = X_raw.astype(float)

    feat_names = [header[i] for i in feat_idx]
    return X, y, feat_names


def train_val_split(X, y, val_ratio=0.2, shuffle=True, random_state=42):
    """
    Chia train/val đơn giản.
    - Có shuffle bằng RNG của NumPy.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    n_val = int(n * val_ratio)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def train_val_test_split(X, y, val_ratio=0.2, test_ratio=0.2,
                         shuffle=True, random_state=42):
    """
    Chia train/val/test:
    - Đầu tiên tách test
    - Phần còn lại chia train/val theo tỉ lệ mong muốn
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    n_test = int(n * test_ratio)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)

    test_idx = idx[:n_test]
    remain = idx[n_test:]
    n_remain = remain.shape[0]
    n_val = int(n_remain * val_ratio / (1.0 - test_ratio))

    val_idx = remain[:n_val]
    train_idx = remain[n_val:]

    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


def accuracy_score(y_true, y_pred):
    """
    Độ chính xác: số mẫu dự đoán đúng / tổng số mẫu.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(y_true == y_pred)


def confusion_matrix_binary(y_true, y_pred, positive_label=1):
    """
    Confusion matrix 2x2 cho bài toán nhị phân:
    [[TN, FP],
     [FN, TP]]
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos = positive_label
    neg = 1 - pos
    TP = np.sum((y_true == pos) & (y_pred == pos))
    TN = np.sum((y_true == neg) & (y_pred == neg))
    FP = np.sum((y_true == neg) & (y_pred == pos))
    FN = np.sum((y_true == pos) & (y_pred == neg))
    return np.array([[TN, FP],
                     [FN, TP]], dtype=int)


def precision_recall_f1(y_true, y_pred, positive_label=1):
    """
    Tính precision, recall, F1 từ confusion matrix.
    - Có bảo vệ chia cho 0.
    """
    cm = confusion_matrix_binary(y_true, y_pred, positive_label)
    TN, FP, FN, TP = cm.ravel()
    precision = 0.0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0.0 if (TP + FN) == 0 else TP / (TP + FN)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_binary_classification(y_true, y_prob, threshold=0.5):
    """
    Gói chung: từ y_true + y_prob → y_pred, tính accuracy, precision, recall, F1, CM.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    cm = confusion_matrix_binary(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def find_best_threshold(y_true, y_prob, n_points=200):
    """
    Quét nhiều threshold trong [0,1], chọn threshold có F1 cao nhất.
    - Loop trên số lượng threshold (ít), bên trong vẫn vector hóa.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    thresholds = np.linspace(0, 1, n_points)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        _, _, f1 = precision_recall_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# ==============================
# LOGISTIC REGRESSION 
class LogisticRegression:
    """
    Cài Logistic Regression từ đầu, hỗ trợ:
    - L1/L2 regularization
    - Optimizer: sgd / momentum / adam
    - Mini-batch
    - LR decay + early stopping
    """

    def __init__(
        self,
        lr=0.1,
        n_epochs=500,
        batch_size=None,
        l2=0.0,
        l1=0.0,
        optimizer="adam",
        lr_decay=0.0,
        early_stopping=False,
        patience=20,
        random_state=42,
    ):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.l1 = l1
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

        self.W = None
        self.train_loss_history = []
        self.val_loss_history = []

    def _add_intercept(self, X):
        """
        Thêm cột bias (1) vào X.
        """
        X = np.asarray(X, float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _init_optimizer(self, dim):
        """
        Khởi tạo các biến động lượng / moment cho optimizer.
        """
        if self.optimizer == "sgd":
            self.m = None
            self.v = None
        elif self.optimizer == "momentum":
            self.m = np.zeros(dim)
            self.v = None
        elif self.optimizer == "adam":
            self.m = np.zeros(dim)
            self.v = np.zeros(dim)
            self.t = 0
        else:
            raise ValueError("optimizer phải là 'sgd', 'momentum' hoặc 'adam'.")

    def _update_weights(self, grad, lr):
        """
        Cập nhật W tuỳ theo optimizer lựa chọn.
        """
        if self.optimizer == "sgd":
            self.W -= lr * grad
        elif self.optimizer == "momentum":
            beta = 0.9
            self.m = beta * self.m + (1 - beta) * grad
            self.W -= lr * self.m
        elif self.optimizer == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            self.t += 1
            self.m = beta1 * self.m + (1 - beta1) * grad
            self.v = beta2 * self.v + (1 - beta2) * (grad * grad)
            # bias-correction
            m_hat = self.m / (1 - beta1 ** self.t)
            v_hat = self.v / (1 - beta2 ** self.t)
            self.W -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Train Logistic Regression với gradient descent + optimizer.
        - Có thể truyền X_val, y_val để theo dõi val_loss + early stopping.
        """
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)

        X_ext = self._add_intercept(X)
        n, d = X_ext.shape

        # Khởi tạo trọng số
        self.W = rng.normal(0, 0.01, size=d)
        self._init_optimizer(d)

        self.train_loss_history = []
        self.val_loss_history = []

        best_loss = np.inf
        wait = 0

        bar = trange(self.n_epochs, desc="LogisticRegression")
        for epoch in bar:
            # Shuffle dữ liệu mỗi epoch
            idx = rng.permutation(n)
            Xs = X_ext[idx]
            ys = y[idx]

            # Chia batch
            if self.batch_size is None:
                batches = [(0, n)]
            else:
                batches = [(i, min(i + self.batch_size, n))
                           for i in range(0, n, self.batch_size)]

            # Duyệt từng batch
            for i, j in batches:
                xb = Xs[i:j]
                yb = ys[i:j]

                z = xb @ self.W
                p = sigmoid(z)
                err = p - yb

                # grad_j = 1/m * sum_i xb_ij * err_i  (vector hóa bằng np.einsum)
                grad = np.einsum("ij,i->j", xb, err) / xb.shape[0]

                # Regularization (không áp dụng cho bias)
                if self.l2 > 0:
                    grad[1:] += self.l2 * self.W[1:]
                if self.l1 > 0:
                    grad[1:] += self.l1 * np.sign(self.W[1:])

                self._update_weights(grad, self.lr)

            # LR decay đơn giản
            if self.lr_decay > 0:
                self.lr *= 1.0 / (1.0 + self.lr_decay * epoch)

            # Loss train
            p_train = sigmoid(X_ext @ self.W)
            train_loss = binary_cross_entropy(y, p_train)
            self.train_loss_history.append(train_loss)

            # Loss val (nếu có)
            val_loss = None
            if X_val is not None and y_val is not None:
                y_val = np.asarray(y_val, float).reshape(-1)
                p_val = self.predict_proba(X_val)
                val_loss = binary_cross_entropy(y_val, p_val)
                self.val_loss_history.append(val_loss)

            monitor = val_loss if val_loss is not None else train_loss
            bar.set_postfix(
                {
                    "train": f"{train_loss:.4f}",
                    "val": "" if val_loss is None else f"{val_loss:.4f}",
                }
            )

            # Early stopping dựa trên monitor (train hoặc val)
            if self.early_stopping:
                if monitor < best_loss - 1e-6:
                    best_loss = monitor
                    wait = 0
                    best_W = self.W.copy()
                else:
                    wait += 1
                if wait >= self.patience:
                    bar.write(
                        f"[Early stopping] quá trình huấn luyện dừng lại tại epoch {epoch + 1} | "
                        f"best_loss = {best_loss:.4f}"
                    )
                    if "best_W" in locals():
                        self.W = best_W
                    break

    def predict_proba(self, X):
        """
        Trả về xác suất P(y=1|x).
        """
        X_ext = self._add_intercept(X)
        return sigmoid(X_ext @ self.W)

    def predict(self, X, threshold=0.5):
        """
        Dự đoán nhãn (0/1) từ xác suất.
        """
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)


# ==============================
# RANDOM FOREST 
def gini_impurity(y):
    """
    Độ đo Gini cho nhãn y.
    """
    y = np.asarray(y).astype(int)
    if y.size == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p * p)


def gini_split(col, y, threshold):
    """
    Gini impurity sau khi tách theo threshold.
    - Dùng cho CART classification.
    """
    col = np.asarray(col)
    y = np.asarray(y).astype(int)

    left = col <= threshold
    right = ~left

    n = len(y)
    n_left = left.sum()
    n_right = right.sum()
    if n_left == 0 or n_right == 0:
        return np.inf

    g_left = gini_impurity(y[left])
    g_right = gini_impurity(y[right])

    return (n_left * g_left + n_right * g_right) / n


class DecisionTree:
    """
    Cây quyết định đơn giản cho phân loại nhị phân:
    - Dùng Gini
    - Tìm split tốt nhất bằng duyệt brute-force features × thresholds
    """

    def __init__(self, max_depth=5, min_samples_split=5, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.pred = None  # dự đoán leaf (đa số class)

    def fit(self, X, y, depth=0):
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)

        # Nếu không còn mẫu: leaf mặc định là 0
        if y.size == 0:
            self.pred = 0
            return

        # Leaf prediction = class xuất hiện nhiều nhất
        counts = np.bincount(y)
        self.pred = counts.argmax()

        # Điều kiện dừng
        if depth >= self.max_depth:
            return
        if y.size < self.min_samples_split:
            return
        if gini_impurity(y) == 0.0:
            return

        n_samples, n_features = X.shape
        best_feat = None
        best_thresh = None
        best_gini = np.inf

        # Duyệt tất cả feature và threshold (CART)
        for f in range(n_features):
            col = X[:, f]
            thresholds = np.unique(col)
            for th in thresholds:
                g = gini_split(col, y, th)
                if g < best_gini:
                    best_gini = g
                    best_feat = f
                    best_thresh = th

        if best_feat is None:
            return

        self.feature = best_feat
        self.threshold = best_thresh

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        self.left = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state + 1,
        )
        self.left.fit(X[left_mask], y[left_mask], depth + 1)

        self.right = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state + 2,
        )
        self.right.fit(X[right_mask], y[right_mask], depth + 1)

    def predict_row(self, x):
        """
        Dự đoán cho 1 mẫu x bằng cách đi theo nhánh của cây.
        """
        if self.feature is None:
            return self.pred
        if x[self.feature] <= self.threshold:
            return self.left.predict_row(x)
        else:
            return self.right.predict_row(x)

    def predict(self, X):
        """
        Dự đoán class cho nhiều mẫu X (loop theo số mẫu, bắt buộc vì cấu trúc cây).
        """
        X = np.asarray(X, float)
        return np.array([self.predict_row(row) for row in X], dtype=int)


class RandomForest:
    """
    Random Forest phân loại nhị phân:
    - Nhiều cây DecisionTree
    - Bootstrap sample trên hàng
    - Subsample feature (max_features)
    - Trung bình dự đoán (dạng xác suất), majority vote cho nhãn.
    """

    def __init__(
        self,
        n_estimators=50,
        max_depth=5,
        max_features="sqrt",
        min_samples_split=5,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.trees = []
        self.feature_subsets = []
        self.train_loss_history = []
        self.val_loss_history = []

    def _sample_features(self, n_features, rng):
        """
        Chọn subset feature cho 1 cây:
        - 'sqrt': sqrt(d)
        - 'all': dùng toàn bộ
        - int: số lượng cụ thể
        """
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "all":
            k = n_features
        elif isinstance(self.max_features, int):
            k = min(n_features, self.max_features)
        else:
            k = n_features
        return rng.choice(n_features, size=k, replace=False)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Huấn luyện RandomForest:
        - Mỗi cây lấy bootstrap sample
        - Mỗi cây chọn subset feature
        - Theo dõi train_loss / val_loss bằng binary_cross_entropy trên output trung bình.
        """
        X = np.asarray(X, float)
        y = np.asarray(y).astype(int)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, float)
            y_val = np.asarray(y_val).astype(int)

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        self.trees = []
        self.feature_subsets = []
        self.train_loss_history = []
        self.val_loss_history = []

        bar = trange(self.n_estimators, desc="RandomForest", leave=True)
        for i in bar:
            # Bootstrap trên hàng
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            Xb = X[idx]
            yb = y[idx]

            # Chọn subset feature
            feats = self._sample_features(n_features, rng)
            self.feature_subsets.append(feats)

            # Train một cây
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state + i,
            )
            tree.fit(Xb[:, feats], yb)
            self.trees.append(tree)

            # Tính xác suất train bằng trung bình dự đoán các cây hiện có
            preds_train = np.array([
                t_.predict(X[:, f_]) for t_, f_ in zip(self.trees, self.feature_subsets)
            ])
            p_train = preds_train.mean(axis=0)
            train_loss = binary_cross_entropy(y, p_train)
            self.train_loss_history.append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                preds_val = np.array([
                    t_.predict(X_val[:, f_]) for t_, f_ in zip(self.trees, self.feature_subsets)
                ])
                p_val = preds_val.mean(axis=0)
                val_loss = binary_cross_entropy(y_val, p_val)
                self.val_loss_history.append(val_loss)

            bar.set_postfix(
                {
                    "train": f"{train_loss:.4f}",
                    "val": "" if val_loss is None else f"{val_loss:.4f}",
                }
            )

    def predict(self, X):
        """
        Majority vote từ các cây để ra nhãn 0/1.
        """
        X = np.asarray(X, float)
        if not self.trees:
            raise ValueError("RandomForest chưa được fit.")

        preds = np.array([
            tree.predict(X[:, feats]) for tree, feats in zip(self.trees, self.feature_subsets)
        ])

        def majority(col):
            counts = np.bincount(col.astype(int))
            return counts.argmax()

        return np.apply_along_axis(majority, 0, preds).astype(int)

    def predict_proba(self, X):
        """
        Xác suất = trung bình dự đoán (0/1) của tất cả cây.
        """
        X = np.asarray(X, float)
        preds = np.array([
            tree.predict(X[:, feats]) for tree, feats in zip(self.trees, self.feature_subsets)
        ])
        return preds.mean(axis=0)


# ==============================
# XGBOOST (PHÂN LOẠI NHỊ PHÂN, LOGISTIC LOSS)
def logistic_grad_hess(y_true, margin):
    """
    Gradient & Hessian của logistic loss:
    - margin = F(x)
    - p = sigmoid(margin)
    - grad = p - y
    - hess = p(1-p)
    """
    y_true = np.asarray(y_true, float).reshape(-1)
    margin = np.asarray(margin, float).reshape(-1)
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h


class RegressionTree:
    """
    Cây Regression dùng trong XGBoost:
    - Mỗi leaf lưu giá trị "weight" (step) để cập nhật F(x).
    - Dùng công thức gain từ G,H (sum grad, hess).
    """

    def __init__(
        self,
        max_depth=3,
        lambda_reg=1.0,
        min_child_weight=1e-3,
        min_gain_to_split=0.0,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.min_child_weight = min_child_weight
        self.min_gain_to_split = min_gain_to_split
        self.random_state = random_state

        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.pred = None  # giá trị leaf

    def _weight(self, G, H):
        """
        Trọng số leaf = -G / (H + lambda) (có epsilon để tránh chia 0).
        """
        return -G / (H + self.lambda_reg + 1e-12)

    def _gain(self, Gl, Hl, Gr, Hr):
        """
        Gain cho một split theo công thức XGBoost (bỏ G,H cha vì không cần so sánh tuyệt đối).
        """
        return 0.5 * (
            (Gl * Gl) / (Hl + self.lambda_reg + 1e-12)
            + (Gr * Gr) / (Hr + self.lambda_reg + 1e-12)
        )

    def fit(self, X, g, h, depth=0):
        """
        Huấn luyện cây hồi quy:
        - Dùng G,H để chọn split tối đa hoá gain.
        """
        X = np.asarray(X, float)
        g = np.asarray(g, float).reshape(-1)
        h = np.asarray(h, float).reshape(-1)

        G = g.sum()
        H = h.sum()
        self.pred = self._weight(G, H)

        # Điều kiện dừng
        if depth >= self.max_depth or X.shape[0] <= 1:
            return

        n_samples, n_features = X.shape
        best_gain = 0.0
        best_feat = None
        best_thresh = None

        # Duyệt brute-force từng feature & threshold
        for f in range(n_features):
            col = X[:, f]
            thresholds = np.unique(col)
            for th in thresholds:
                left = col <= th
                right = ~left

                if left.sum() == 0 or right.sum() == 0:
                    continue

                Gl = g[left].sum()
                Hl = h[left].sum()
                Gr = g[right].sum()
                Hr = h[right].sum()

                # Không chia tiếp nếu child quá "nhẹ"
                if Hl < self.min_child_weight or Hr < self.min_child_weight:
                    continue

                gain = self._gain(Gl, Hl, Gr, Hr)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = th

        # Nếu không có split nào tốt hơn min_gain_to_split → leaf
        if best_feat is None or best_gain < self.min_gain_to_split:
            return

        self.feature = best_feat
        self.threshold = best_thresh

        left = X[:, best_feat] <= best_thresh
        right = ~left

        self.left = RegressionTree(
            max_depth=self.max_depth,
            lambda_reg=self.lambda_reg,
            min_child_weight=self.min_child_weight,
            min_gain_to_split=self.min_gain_to_split,
            random_state=self.random_state + 1,
        )
        self.left.fit(X[left], g[left], h[left], depth + 1)

        self.right = RegressionTree(
            max_depth=self.max_depth,
            lambda_reg=self.lambda_reg,
            min_child_weight=self.min_child_weight,
            min_gain_to_split=self.min_gain_to_split,
            random_state=self.random_state + 2,
        )
        self.right.fit(X[right], g[right], h[right], depth + 1)

    def predict_row(self, x):
        """
        Dự đoán margin cho một mẫu x (đi theo cây đến leaf).
        """
        if self.feature is None:
            return self.pred
        if x[self.feature] <= self.threshold:
            return self.left.predict_row(x)
        else:
            return self.right.predict_row(x)

    def predict(self, X):
        """
        Dự đoán margin cho nhiều mẫu X.
        """
        X = np.asarray(X, float)
        return np.array([self.predict_row(row) for row in X], dtype=float)


class XGBoost:
    """
    Cài đặt XGBoost đơn giản cho bài toán nhị phân (logistic loss):
    - Huấn luyện nhiều RegressionTree tuần tự
    - F(t+1) = F(t) + lr * tree_t(x)
    - Có subsample hàng, colsample_by_tree, early stopping.
    """

    def __init__(
        self,
        n_rounds=50,
        max_depth=3,
        learning_rate=0.1,
        lambda_reg=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1e-3,
        min_gain_to_split=0.0,
        early_stopping=False,
        patience=20,
        random_state=42,
    ):
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.min_gain_to_split = min_gain_to_split
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state

        self.trees = []
        self.feature_subsets = []
        self.train_loss_history = []
        self.val_loss_history = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Huấn luyện XGBoost:
        - Xây dần F(x) qua nhiều round.
        - Mỗi round:
          + Tính grad/hess
          + Subsample hàng & cột
          + Train RegressionTree
          + Cập nhật F, tính loss, kiểm tra early stopping.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1)
        n_samples, n_features = X.shape

        self.trees = []
        self.feature_subsets = []
        self.train_loss_history = []
        self.val_loss_history = []

        # F(x) ban đầu = 0
        F = np.zeros(n_samples, float)

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, float)
            y_val = np.asarray(y_val, float).reshape(-1)
            F_val = np.zeros(X_val.shape[0], float)
        else:
            F_val = None

        rng = np.random.default_rng(self.random_state)
        best_loss = np.inf
        wait = 0

        bar = trange(self.n_rounds, desc="XGBoost")
        for t in bar:
            # Gradient + Hessian logistic
            g, h = logistic_grad_hess(y, F)

            # Subsample hàng
            if self.subsample < 1.0:
                m = max(1, int(self.subsample * n_samples))
                idx_row = rng.choice(n_samples, size=m, replace=False)
            else:
                idx_row = np.arange(n_samples)

            # Subsample cột
            if self.colsample_bytree < 1.0:
                k = max(1, int(self.colsample_bytree * n_features))
                idx_col = rng.choice(n_features, size=k, replace=False)
            else:
                idx_col = np.arange(n_features)

            X_sub = X[idx_row][:, idx_col]
            g_sub = g[idx_row]
            h_sub = h[idx_row]

            # Train một RegressionTree trên (X_sub, g_sub, h_sub)
            tree = RegressionTree(
                max_depth=self.max_depth,
                lambda_reg=self.lambda_reg,
                min_child_weight=self.min_child_weight,
                min_gain_to_split=self.min_gain_to_split,
                random_state=self.random_state + t,
            )
            tree.fit(X_sub, g_sub, h_sub, depth=0)

            self.trees.append(tree)
            self.feature_subsets.append(idx_col)

            # Cập nhật F(x) trên toàn bộ X
            F += self.learning_rate * tree.predict(X[:, idx_col])

            p_train = sigmoid(F)
            train_loss = binary_cross_entropy(y, p_train)
            self.train_loss_history.append(train_loss)

            val_loss = None
            if X_val is not None:
                F_val += self.learning_rate * tree.predict(X_val[:, idx_col])
                p_val = sigmoid(F_val)
                val_loss = binary_cross_entropy(y_val, p_val)
                self.val_loss_history.append(val_loss)

            monitor = val_loss if val_loss is not None else train_loss
            bar.set_postfix(
                {
                    "train": f"{train_loss:.4f}",
                    "val": "" if val_loss is None else f"{val_loss:.4f}",
                }
            )

            # Early stopping dựa trên val_loss (nếu có)
            if self.early_stopping:
                if monitor < best_loss - 1e-6:
                    best_loss = monitor
                    wait = 0
                    best_t = len(self.trees)
                else:
                    wait += 1
                if wait >= self.patience:
                    bar.write(
                        f"[Early stopping] XGBoost dừng tại round {t + 1} | "
                        f"best_val_loss = {best_loss:.4f}, "
                        f"số cây dùng thực tế = {best_t}"
                    )
                    if "best_t" in locals():
                        self.trees = self.trees[:best_t]
                        self.feature_subsets = self.feature_subsets[:best_t]
                    break

    def _predict_margin(self, X):
        """
        Trả về margin F(x) = sum_t lr * f_t(x).
        """
        X = np.asarray(X, float)
        n_samples = X.shape[0]
        F = np.zeros(n_samples, float)
        for tree, feats in zip(self.trees, self.feature_subsets):
            F += self.learning_rate * tree.predict(X[:, feats])
        return F

    def predict_proba(self, X):
        """
        Xác suất = sigmoid(F(x)).
        """
        F = self._predict_margin(X)
        return sigmoid(F)

    def predict(self, X, threshold=0.5):
        """
        Dự đoán nhãn 0/1 từ xác suất.
        """
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
