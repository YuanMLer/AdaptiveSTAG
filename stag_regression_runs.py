"""
STAG (Bounded Asymmetric Gaussian) Regression Model - 30 Runs Average
For SECOM dataset failure rate prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request
from scipy.stats import skew

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def download_and_preprocess_secom(data_dir="./secom_data"):
    """
    Download and preprocess SECOM dataset
    :param data_dir: data storage directory
    :return: standardized features X, failure rate y via sliding window
    """
    os.makedirs(data_dir, exist_ok=True)

    # UCI SECOM dataset URLs
    secom_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    secom_labels_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"

    secom_data_path = os.path.join(data_dir, "secom.data")
    secom_labels_path = os.path.join(data_dir, "secom_labels.data")

    # Download dataset (first run)
    if not os.path.exists(secom_data_path):
        print("Downloading SECOM dataset...")
        urllib.request.urlretrieve(secom_data_url, secom_data_path)
        urllib.request.urlretrieve(secom_labels_url, secom_labels_path)
        print("Download complete!")

    # Data loading and preprocessing
    print("Preprocessing SECOM dataset...")
    data = pd.read_csv(secom_data_path, sep=" ", header=None)
    labels = pd.read_csv(secom_labels_path, sep=" ", header=None)

    # Missing value imputation (mean)
    data = data.fillna(data.mean())

    # Feature standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(data.values)

    # Sliding window for local failure rate (window size=50)
    window_size = 50
    y = []
    for i in range(len(labels)):
        start = max(0, i - window_size // 2)
        end = min(len(labels), i + window_size // 2 + 1)
        local_labels = (labels[0][start:end].values + 1) / 2  # Convert labels from {-1,1} to {0,1}
        local_fail_rate = local_labels.mean()
        y.append(local_fail_rate)
    y = np.array(y).reshape(-1, 1)

    # Print data statistics
    print(f"Preprocessing complete! Data size: {len(X)}, Feature dim: {X.shape[1]}")
    print(f"Failure rate stats: mean={y.mean():.4f}, median={np.median(y):.4f}, 90th percentile={np.percentile(y, 90):.4f}")

    return X, y


def gaussian_cdf(x):
    """Gaussian CDF"""
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gaussian_icdf(p):
    """Gaussian inverse CDF"""
    return torch.erfinv(2.0 * p - 1.0) * math.sqrt(2.0)


class SECOMDataset(Dataset):
    """SECOM dataset loader"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GaussianRegressor(nn.Module):
    """Standard Gaussian regression model"""
    def __init__(self, input_dim=590, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, 1)
        self.fc_logvar = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-6, 2)  # Clamp variance to avoid explosion
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        y_pred = mu + std * eps
        return y_pred, mu, std

    def compute_loss(self, y_pred, y_true, mu, std):
        """Negative log-likelihood loss"""
        loss = 0.5 * torch.sum(((y_true - mu) / std) ** 2 + torch.log(std ** 2) + math.log(2 * math.pi))
        return loss


class TruncatedGaussianRegressor(nn.Module):
    """Truncated Gaussian regression model (0-1 interval)"""
    def __init__(self, input_dim=590, hidden_dim=128, a=0.0, b=1.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, 1)
        self.fc_logsigma = nn.Linear(hidden_dim // 2, 1)
        self.a = a  # lower bound
        self.b = b  # upper bound

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h).clamp(self.a, self.b)
        logsigma = self.fc_logsigma(h).clamp(-6, 2)
        sigma = torch.exp(logsigma) + 1e-6  # Avoid sigma=0

        # Reparameterization sampling (truncated Gaussian)
        eps = 1e-6
        alpha = (self.a - mu) / sigma
        beta = (self.b - mu) / sigma
        u = torch.rand_like(mu).clamp(eps, 1.0 - eps)
        cdf_alpha = gaussian_cdf(alpha)
        cdf_beta = gaussian_cdf(beta)
        p = cdf_alpha + u * (cdf_beta - cdf_alpha)
        z_std = gaussian_icdf(p)
        y_pred = mu + sigma * z_std
        return y_pred.clamp(self.a, self.b), mu, sigma

    def compute_loss(self, y_pred, y_true, mu, sigma):
        """Truncated Gaussian NLL loss"""
        Z = sigma * math.sqrt(2.0 * math.pi) * (gaussian_cdf((self.b - mu)/sigma) - gaussian_cdf((self.a - mu)/sigma))
        log_pdf = -((y_true - mu) ** 2) / (2.0 * sigma ** 2) - torch.log(Z)
        loss = -torch.sum(log_pdf)
        return loss


def reparam_ag_fixed(mu, sigma_L, sigma_R):
    """Asymmetric Gaussian reparameterization sampling"""
    p_L = sigma_L / (sigma_L + sigma_R)
    u = torch.rand_like(mu).clamp(1e-8, 1.0 - 1e-8)
    z = torch.zeros_like(mu)

    # Left side sampling
    left_mask = u <= p_L
    if left_mask.any():
        u_left = u[left_mask] / p_L[left_mask]
        m = mu[left_mask]
        sl = sigma_L[left_mask]
        target = 0.5 - u_left / 2.0
        x_std = gaussian_icdf(target)
        z[left_mask] = m + sl * x_std

    # Right side sampling
    right_mask = ~left_mask
    if right_mask.any():
        u_right = (u[right_mask] - p_L[right_mask]) / (1.0 - p_L[right_mask])
        m = mu[right_mask]
        sr = sigma_R[right_mask]
        target = 0.5 + u_right / 2.0
        x_std = gaussian_icdf(target)
        z[right_mask] = m + sr * x_std

    return z


class AGRegressor(nn.Module):
    """Asymmetric Gaussian (AG) regression model"""
    def __init__(self, input_dim=590, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_sl = nn.Linear(hidden_dim // 2, 1)  # Left std log
        self.fc_log_sr = nn.Linear(hidden_dim // 2, 1)  # Right std log

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_sl = self.fc_log_sl(h).clamp(-6, 2)
        log_sr = self.fc_log_sr(h).clamp(-6, 2)
        sl = torch.exp(log_sl) + 1e-6
        sr = torch.exp(log_sr) + 1e-6
        y_pred = reparam_ag_fixed(mu, sl, sr)
        return y_pred, mu, sl, sr

    def compute_loss(self, y_pred, y_true, mu, sl, sr):
        """Asymmetric Gaussian NLL loss"""
        sigma = torch.where(y_true < mu, sl, sr)
        Z = math.sqrt(2.0 * math.pi) * (sl + sr) / 2.0
        log_pdf = -((y_true - mu) ** 2) / (2.0 * sigma ** 2) - torch.log(Z)
        loss = -torch.sum(log_pdf)
        return loss


def stag_norm_const(mu, sigma_L, sigma_R, a, b):
    """STAG distribution normalization constant"""
    term_L = sigma_L * (gaussian_cdf((mu - a) / sigma_L) - 0.5)
    term_R = sigma_R * (gaussian_cdf((b - mu) / sigma_R) - 0.5)
    Z = math.sqrt(2.0 * math.pi) * (term_L + term_R)
    return Z


def reparam_stag(mu, sigma_L, sigma_R, a=0.0, b=1.0):
    """STAG distribution reparameterization sampling"""
    Z = stag_norm_const(mu, sigma_L, sigma_R, a, b)
    term_L = sigma_L * (gaussian_cdf((mu - a) / sigma_L) - 0.5)
    p_L = (math.sqrt(2.0 * math.pi) * term_L) / Z
    eps = 1e-6
    u = torch.rand_like(mu).clamp(eps, 1.0 - eps)
    z = torch.zeros_like(mu)
    
    # Left interval sampling
    left_mask = u <= p_L
    if left_mask.any():
        u_left = u[left_mask] / p_L[left_mask]
        u_left = u_left.clamp(eps, 1.0 - eps)
        m = mu[left_mask]
        sl = sigma_L[left_mask]
        Phi_ma = gaussian_cdf((m - a) / sl)
        target = Phi_ma - u_left * (Phi_ma - 0.5)
        target = target.clamp(eps, 1.0 - eps)
        x_std = gaussian_icdf(target)
        z[left_mask] = m - sl * x_std
    
    # Right interval sampling
    right_mask = ~left_mask
    if right_mask.any():
        u_right = (u[right_mask] - p_L[right_mask]) / (1.0 - p_L[right_mask])
        u_right = u_right.clamp(eps, 1.0 - eps)
        m = mu[right_mask]
        sr = sigma_R[right_mask]
        Phi_bm = gaussian_cdf((b - m) / sr)
        target = 0.5 + u_right * (Phi_bm - 0.5)
        target = target.clamp(eps, 1.0 - eps)
        x_std = gaussian_icdf(target)
        z[right_mask] = m + sr * x_std
    
    return z.clamp(a, b)


class STAGRegressor(nn.Module):
    """Adaptive boundary STAG regression model (core model)"""
    def __init__(self, input_dim=590, hidden_dim=128, bound_reg=0.0001):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_sl = nn.Linear(hidden_dim // 2, 1)
        self.fc_log_sr = nn.Linear(hidden_dim // 2, 1)

        # Boundary regularization coefficient
        self.bound_reg = bound_reg
        # Adaptive boundary parameters (mapped via sigmoid to [-0.1,0.1] and [0.9,1.1])
        self.a_raw = nn.Parameter(torch.tensor(0.0))
        self.b_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_sl = self.fc_log_sl(h).clamp(-6, 2)
        log_sr = self.fc_log_sr(h).clamp(-6, 2)
        sl = torch.exp(log_sl) + 1e-6
        sr = torch.exp(log_sr) + 1e-6

        # Adaptive boundary computation (sigmoid mapping)
        a = -0.1 + torch.sigmoid(self.a_raw) * 0.2
        b = 0.9 + torch.sigmoid(self.b_raw) * 0.2

        mu = mu.clamp(a, b)
        y_pred = reparam_stag(mu, sl, sr, a, b)
        return y_pred, mu, sl, sr

    def compute_loss(self, y_pred, y_true, mu, sl, sr):
        """STAG loss with boundary regularization"""
        # Compute current adaptive boundaries
        a = -0.1 + torch.sigmoid(self.a_raw) * 0.2
        b = 0.9 + torch.sigmoid(self.b_raw) * 0.2

        # Negative log-likelihood loss
        sigma = torch.where(y_true < mu, sl, sr)
        Z = stag_norm_const(mu, sl, sr, a, b)
        log_pdf = -((y_true - mu) ** 2) / (2.0 * sigma ** 2) - torch.log(Z)
        nll_loss = -torch.sum(log_pdf)

        # Boundary regularization loss (constrain a->0, b->1)
        bound_reg_loss = self.bound_reg * (a**2 + (b - 1.0)**2)
        return nll_loss + bound_reg_loss


def train_single_model(model, train_loader, test_loader, epochs, device):
    """
    Train single model and evaluate
    :param model: model to train
    :param train_loader: training data loader
    :param test_loader: test data loader
    :param epochs: training epochs
    :param device: training device (CPU/GPU)
    :return: avg MSE, boundary error rate, true values, predicted values
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # Learning rate decay

    all_y_true = []
    all_y_pred = []

    # Training phase
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward and loss computation for different models
            if isinstance(model, GaussianRegressor):
                y_pred, mu, std = model(x)
                loss = model.compute_loss(y_pred, y, mu, std)
            elif isinstance(model, TruncatedGaussianRegressor):
                y_pred, mu, sigma = model(x)
                loss = model.compute_loss(y_pred, y, mu, sigma)
            elif isinstance(model, (AGRegressor, STAGRegressor)):
                y_pred, mu, sl, sr = model(x)
                loss = model.compute_loss(y_pred, y, mu, sl, sr)
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

            loss.backward()
            optimizer.step()

        scheduler.step()

    # Testing phase
    model.eval()
    total_mse = 0
    total_boundary_error = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # Forward to get predictions
            if isinstance(model, (GaussianRegressor, TruncatedGaussianRegressor)):
                y_pred, _, _ = model(x)
            else:
                y_pred, _, _, _ = model(x)

            # Compute MSE and boundary error (predictions outside 0-1)
            mse = F.mse_loss(y_pred, y, reduction='sum')
            total_mse += mse.item()
            total_boundary_error += torch.sum((y_pred < 0.0) | (y_pred > 1.0)).item()

            all_y_true.append(y.cpu().numpy())
            all_y_pred.append(y_pred.cpu().numpy())

    # Concatenate all predictions
    all_y_true = np.concatenate(all_y_true).flatten()
    all_y_pred = np.concatenate(all_y_pred).flatten()

    # Compute average metrics
    avg_mse = total_mse / len(test_loader.dataset)
    avg_boundary_error = total_boundary_error / len(test_loader.dataset)

    return avg_mse, avg_boundary_error, all_y_true, all_y_pred


def run_single_experiment(X, y, random_state, device, epochs):
    """
    Single experiment (split train/test, train and evaluate all models)
    :param X: feature matrix
    :param y: label vector
    :param random_state: random seed (for reproducibility)
    :param device: training device
    :param epochs: training epochs
    :return: dict of model results, test labels
    """
    # Split train/test (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Build data loaders
    train_dataset = SECOMDataset(X_train, y_train)
    test_dataset = SECOMDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    results = {}

    # 1. Standard Gaussian model
    model_gauss = GaussianRegressor(input_dim=X.shape[1])
    mse, be, y_true, y_pred = train_single_model(model_gauss, train_loader, test_loader, epochs, device)
    results['Gaussian'] = {'mse': mse, 'be': be, 'y_true': y_true, 'y_pred': y_pred}

    # 2. Truncated Gaussian model
    model_trunc = TruncatedGaussianRegressor(input_dim=X.shape[1], a=0.0, b=1.0)
    mse, be, y_true, y_pred = train_single_model(model_trunc, train_loader, test_loader, epochs, device)
    results['TruncatedGaussian'] = {'mse': mse, 'be': be, 'y_true': y_true, 'y_pred': y_pred}

    # 3. Asymmetric Gaussian model
    model_ag = AGRegressor(input_dim=X.shape[1])
    mse, be, y_true, y_pred = train_single_model(model_ag, train_loader, test_loader, epochs, device)
    results['AG'] = {'mse': mse, 'be': be, 'y_true': y_true, 'y_pred': y_pred}

    # 4. STAG model (core model)
    model_stag = STAGRegressor(input_dim=X.shape[1], bound_reg=0.0001)
    mse, be, y_true, y_pred = train_single_model(model_stag, train_loader, test_loader, epochs, device)
    results['STAG'] = {'mse': mse, 'be': be, 'y_true': y_true, 'y_pred': y_pred}

    return results, y_test.flatten()


if __name__ == "__main__":
    # ===================== Global fixed random seeds =====================
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ====================================================================

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Experiment hyperparameters
    epochs = 100
    n_runs = 30  # 30 runs average (paper requirement)

    # Experiment info
    print("=" * 80)
    print(f"SECOM Experiment - {n_runs} runs average")
    print("=" * 80)

    # Data preprocessing
    X, y = download_and_preprocess_secom()

    # Initialize result storage
    print(f"\nStarting {n_runs} experiments...")
    all_results = {
        name: {'mse': [], 'be': [], 'y_true': [], 'y_pred': []} 
        for name in ['Gaussian', 'TruncatedGaussian', 'AG', 'STAG']
    }
    all_y_test = []

    # Run multiple experiments
    for run_idx in range(n_runs):
        # Print progress (every 10 runs)
        if (run_idx + 1) % 10 == 0:
            print(f"Progress: {run_idx + 1}/{n_runs}")

        # Single experiment
        results, y_test = run_single_experiment(
            X, y, random_state=run_idx, device=device, epochs=epochs
        )
        all_y_test.append(y_test)

        # Collect results
        for name, res in results.items():
            all_results[name]['mse'].append(res['mse'])
            all_results[name]['be'].append(res['be'])
            all_results[name]['y_true'].append(res['y_true'])
            all_results[name]['y_pred'].append(res['y_pred'])

    # Results summary
    print("\n" + "=" * 80)
    print(f"Experiment Results Summary ({n_runs} runs average)")
    print("=" * 80)

    # Print quantitative results (for paper tables)
    print(f"\n{'Model':<20} {'Test MSE':>12} {'Std':>10} {'Boundary Err':>14} {'Std':>10}")
    print("-" * 70)

    for name in ['Gaussian', 'TruncatedGaussian', 'AG', 'STAG']:
        mse_list = all_results[name]['mse']
        be_list = [b * 100 for b in all_results[name]['be']]  # Convert to percentage

        # Compute mean and std
        mse_mean = np.mean(mse_list)
        mse_std = np.std(mse_list)
        be_mean = np.mean(be_list)
        be_std = np.std(be_list)

        # Format print
        print(f"{name:<20} {mse_mean:>12.6f} {mse_std:>10.6f} {be_mean:>13.2f}% {be_std:>9.2f}%")

    print("=" * 80)