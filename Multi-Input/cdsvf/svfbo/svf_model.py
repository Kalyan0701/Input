"""
Task 1 (facility-level embeddings) and Task 2 (single-view fusion), proposal notation.

Task 1, per facility l:   z = F_en(x),   x~ = F_de(z),   z in R^q
    NNAutoencoder : the proposal's NNAE (needs more data)
    PCAEncoder    : the optimal LINEAR autoencoder under squared error; closed form, no
                    training, deterministic. Small-data option. (It is also the linear-kernel
                    special case of GP latent-variable models, the family MoGPAE belongs to.)

Task 2, all facilities, proposal Eqs. (1)-(4):
    h     = u_l(z)          one encoder per facility
    z~    = v_l(h)          one decoder per facility
    y^    = w(h)            ONE predictor shared by all facilities
    loss  = sum_l [ mean ||z - v_l(u_l(z))||^2  +  lam * mean (y - w(u_l(z)))^2 ]

Fixes relative to the old svf_model.py:
  * y is kept as an (n, 1) column and shapes are asserted, so MSE can never broadcast
    to an n x n matrix again (that made the prediction term learn only the mean of y).
  * w takes h only (predictor_uses_z=True restores w([h, z]) as an ablation).
  * Full-batch training; models can be warm-started across BO iterations.
"""
import numpy as np
import torch
import torch.nn as nn


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


def _t(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32)


# --------------------------------------------------------------------------- Task 1
class PCAEncoder:
    def __init__(self, q):
        self.q = q
        self.mean = None
        self.V = None  # (d, q) principal directions

    def fit(self, X, epochs=None):  # `epochs` ignored; same signature as the NN version
        mean = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
        V = Vt[: self.q].T
        if V.shape[1] < self.q:  # fewer points than q: pad with zero directions
            V = np.hstack([V, np.zeros((V.shape[0], self.q - V.shape[1]))])
        if self.V is not None:   # SVD signs are arbitrary; keep them consistent across refits
            signs = np.sign(np.sum(V * self.V, axis=0))
            signs[signs == 0] = 1.0
            V = V * signs
        self.mean, self.V = mean, V
        return self

    def encode(self, X):
        return (np.atleast_2d(X) - self.mean) @ self.V

    def decode(self, Z):
        return np.atleast_2d(Z) @ self.V.T + self.mean


class NNAutoencoder:
    def __init__(self, d, q, hidden, lr, weight_decay):
        self.enc = mlp(d, hidden, q)
        self.dec = mlp(q, hidden, d)
        params = list(self.enc.parameters()) + list(self.dec.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.last_loss = None

    def fit(self, X, epochs):
        Xt = _t(X)
        for _ in range(epochs):
            self.opt.zero_grad()
            loss = ((self.dec(self.enc(Xt)) - Xt) ** 2).mean()
            loss.backward()
            self.opt.step()
        self.last_loss = float(loss)
        return self

    @torch.no_grad()
    def encode(self, X):
        return self.enc(_t(np.atleast_2d(X))).numpy().astype(float)

    @torch.no_grad()
    def decode(self, Z):
        return self.dec(_t(np.atleast_2d(Z))).numpy().astype(float)


# --------------------------------------------------------------------------- Task 2
class SVFNet(nn.Module):
    def __init__(self, L, q, h_dim, hidden, predictor_uses_z=False):
        super().__init__()
        self.u = nn.ModuleList([mlp(q, hidden, h_dim) for _ in range(L)])
        self.v = nn.ModuleList([mlp(h_dim, hidden, q) for _ in range(L)])
        self.predictor_uses_z = predictor_uses_z
        self.w = mlp(h_dim + (q if predictor_uses_z else 0), hidden, 1)

    def encode(self, z, l):
        return self.u[l](z)

    def decode(self, h, l):
        return self.v[l](h)

    def predict(self, h, z=None):
        x = torch.cat([h, z], dim=1) if self.predictor_uses_z else h
        return self.w(x)


def train_svf(net, Z_list, Y_list, lam, epochs, opt):
    """
    Full-batch minimization of Eq. (4).
    Z_list[l]: (n_l, q) standardized z of facility l.
    Y_list[l]: (n_l,)   y of facility l, standardized with the SHARED y transform.
    Returns (reconstruction, prediction) loss every 100 epochs and at the end.
    """
    Zt = [_t(Z) for Z in Z_list]
    Yt = [_t(Y).reshape(-1, 1) for Y in Y_list]  # column vector: the fix for the broadcast bug
    history = []
    for epoch in range(epochs):
        opt.zero_grad()
        rec, pred = 0.0, 0.0
        for l, (z, y) in enumerate(zip(Zt, Yt)):
            h = net.encode(z, l)
            z_hat = net.decode(h, l)
            y_hat = net.predict(h, z)
            assert y_hat.shape == y.shape, f"shape mismatch {tuple(y_hat.shape)} vs {tuple(y.shape)}"
            rec = rec + ((z_hat - z) ** 2).mean()
            pred = pred + ((y_hat - y) ** 2).mean()
        loss = rec + lam * pred
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == epochs - 1:
            history.append((float(rec), float(pred)))
    return history