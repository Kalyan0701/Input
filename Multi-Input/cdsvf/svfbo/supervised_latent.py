"""
Supervised shared latent (the recommended method; feasibility version).

Per facility l, an affine encoder maps the box-normalized design to the shared space:
    z = V_l^T (x01 - m_l)          projection onto the span of the facility's own designs
    h = W_l z + b_l,               W_l in R^{k x p_l}, p_l = rank of the facility's designs
(The projection is exact PCA keeping every non-negligible direction, i.e. Task 1 without
compression; the supervised map z -> h plays the role of Task 2.)
One GP over ALL facilities' (h, y):
    y = f(h) + noise,  f ~ GP(0, s^2 exp(-||h - h'||^2 / 2))
The encoders and the GP hyperparameters are trained TOGETHER by maximizing the pooled GP
log marginal likelihood (plus a small Gaussian prior on W). So the directions in h are
chosen by y, not by how the designs happen to vary (which is what Task 1 does).

The kernel length scale is fixed at 1: the encoder's scale already plays that role.
Gradients are analytic, so this runs in plain numpy/scipy (no torch needed).

Math used for the gradient (n pooled points, K = s^2 R + noise I, R_ij = exp(-||h_i-h_j||^2/2)):
    NLL      = 1/2 y^T K^-1 y + 1/2 log|K| + n/2 log(2 pi)
    dNLL/dK  = M = 1/2 (K^-1 - a a^T),   a = K^-1 y
    dNLL/dh_i = -2 sum_j M_ij s^2 R_ij (h_i - h_j)
    dNLL/dW_l = G_l^T Z_l,  dNLL/db_l = sum of rows of G_l   (G = dNLL/dH)
Optimization: L-BFGS with noise annealing (fixed noise 1.0 -> 0.01, then learned) and restarts;
without annealing L-BFGS stalls far above the loss of the true encoder.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .normalization import Standardizer


class SupervisedLatent:
    def __init__(self, d_list, h_dim=5, prior=1e-2, restarts=3, maxiter=150, seed=0,
                 rank_tol=5e-3, anneal=(1.0, 0.3, 0.1, 0.03, 0.01)):
        self.d_list = list(d_list)
        self.k = h_dim
        self.prior = prior
        self.restarts = restarts
        self.maxiter = maxiter
        self.seed = seed
        self.rank_tol = rank_tol   # keep principal directions with singular value > tol * largest
        self.anneal = anneal       # fixed noise levels optimized in turn, then noise is learned

    # ---------------------------------------------------------------- parameters
    def _unpack(self, theta):
        W, b, i = [], [], 0
        for d in self.d_list:
            W.append(theta[i:i + self.k * d].reshape(self.k, d)); i += self.k * d
            b.append(theta[i:i + self.k]); i += self.k
        return W, b, theta[i], theta[i + 1]          # log s^2, raw noise

    def _pack(self, W, b, log_s2, raw_noise):
        parts = []
        for Wl, bl in zip(W, b):
            parts += [Wl.ravel(), bl]
        return np.concatenate(parts + [[log_s2, raw_noise]])

    # ---------------------------------------------------------------- objective
    def _loss_grad(self, theta):
        W, b, log_s2, raw_noise = self._unpack(theta)
        H = np.vstack([X @ Wl.T + bl for X, Wl, bl in zip(self.X, W, b)])
        n = H.shape[0]
        s2 = np.exp(log_s2)
        noise = 1e-4 + np.exp(raw_noise)
        D2 = np.sum((H[:, None, :] - H[None, :, :]) ** 2, axis=-1)
        R = np.exp(-0.5 * D2)
        K = s2 * R + noise * np.eye(n)
        try:
            cf = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros_like(theta)
        a = cho_solve(cf, self.y)
        Kinv = cho_solve(cf, np.eye(n))
        nll = 0.5 * self.y @ a + np.sum(np.log(np.diag(cf[0]))) + 0.5 * n * np.log(2 * np.pi)
        M = 0.5 * (Kinv - np.outer(a, a))

        P = M * (s2 * R)                                   # elementwise
        G = -2.0 * (P.sum(axis=1)[:, None] * H - P @ H)    # dNLL/dH
        gW, gb, i = [], [], 0
        loss = nll
        for X, Wl in zip(self.X, W):
            Gl = G[i:i + X.shape[0]]; i += X.shape[0]
            gW.append(Gl.T @ X + self.prior * Wl)
            gb.append(Gl.sum(axis=0))
            loss += 0.5 * self.prior * np.sum(Wl ** 2)
        g_log_s2 = np.sum(M * s2 * R)
        g_raw = np.trace(M) * np.exp(raw_noise)
        return loss, self._pack(gW, gb, g_log_s2, g_raw)

    # ---------------------------------------------------------------- init
    def _init(self, rng, pca):
        W, b = [], []
        for X in self.X:
            Xc = X - X.mean(axis=0)
            if pca:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                V = Vt[: self.k]
                if V.shape[0] < self.k:  # fewer points than k
                    V = np.vstack([V, rng.standard_normal((self.k - V.shape[0], X.shape[1]))])
            else:
                V = rng.standard_normal((self.k, X.shape[1]))
            Hs = Xc @ V.T
            V = V / np.maximum(Hs.std(axis=0), 1e-6)[:, None]   # unit spread per h-dimension
            W.append(V)
            b.append(-X.mean(axis=0) @ V.T)
        return self._pack(W, b, 0.0, np.log(0.01))

    # ---------------------------------------------------------------- public
    def fit(self, X01_list, y_list):
        # Restrict each encoder to the span of its facility's designs: the data never move in
        # the other directions, so nothing can be learned there (and fitting y through the tiny
        # noise in those directions blows up on new designs).
        self.proj, Z = [], []
        for X in X01_list:
            X = np.asarray(X, float)
            mean = X.mean(axis=0)
            _, sv, Vt = np.linalg.svd(X - mean, full_matrices=False)
            V = Vt[sv > self.rank_tol * sv[0]].T            # (d_l, p_l)
            self.proj.append((mean, V))
            Z.append((X - mean) @ V)
        self.d_list = [z.shape[1] for z in Z]
        self.X = Z
        self.y_std = Standardizer().fit(np.concatenate(y_list))
        self.y = self.y_std.transform(np.concatenate(y_list))

        rng = np.random.default_rng([self.seed, 31])
        best = None
        for r in range(self.restarts):
            theta = self._init(rng, pca=(r == 0))
            # Noise annealing: a large fixed noise smooths the landscape; shrink it step by step.
            for noise in self.anneal:
                raw = np.log(max(noise - 1e-4, 1e-8))

                def f(t, raw=raw):
                    loss, g = self._loss_grad(np.concatenate([t, [raw]]))
                    return loss, g[:-1]
                res = minimize(f, theta[:-1], jac=True, method="L-BFGS-B",
                               options={"maxiter": self.maxiter})
                theta = np.concatenate([res.x, [raw]])
            res = minimize(self._loss_grad, theta, jac=True, method="L-BFGS-B",
                           options={"maxiter": self.maxiter})
            if best is None or res.fun < best.fun:
                best = res
        self.W, self.b, self.log_s2, self.raw_noise = self._unpack(best.x)
        self.final_loss = float(best.fun)
        return self

    def project(self, x01, l):
        mean, V = self.proj[l]
        return (np.atleast_2d(x01) - mean) @ V

    def encode(self, x01, l):
        return self.project(x01, l) @ self.W[l].T + self.b[l]