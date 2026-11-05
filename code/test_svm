"""
Dual soft-margin SVM, four kernels (Math 582 mini project)

  min_a 0.5 a^T Q a - 1^T a,   Q_ij = y_i y_j K(x_i, x_j)
  s.t.  0 <= a_i <= C,  y^T a = 0

Kernels
  linear   K = x^T x'
  poly     K = (x^T x' + r)^d
  rbf      K = exp(-gamma ||x - x'||^2)
  sigmoid  K = tanh(gamma x^T x' + r)      (not positive semidefinite)
"""
import os
import time

import numpy as np
from cvxopt import matrix, solvers, spmatrix
from scipy.optimize import minimize


# ------------------------------------------------------------------ kernels
def kernel_matrix(X1, X2, kernel, gamma=1.0, r=1.0, degree=3):
    if kernel == "linear":
        return X1 @ X2.T
    if kernel == "poly":
        return (X1 @ X2.T + r) ** degree
    if kernel == "rbf":
        sq = (X1 ** 2).sum(1)[:, None] + (X2 ** 2).sum(1)[None, :] - 2 * X1 @ X2.T
        return np.exp(-gamma * np.maximum(sq, 0.0))       # clip round-off negatives
    if kernel == "sigmoid":
        return np.tanh(gamma * (X1 @ X2.T) + r)
    raise ValueError(f"unknown kernel {kernel}")


# ------------------------------------------------------------------ model
class DualSVM:
    def __init__(self, kernel="rbf", C=1.0, gamma="scale", r=1.0, degree=3,
                 solver="cvxopt", class_weight=None, sv_tol=1e-6, ridge=1e-8):
        self.kernel, self.C, self.gamma, self.r, self.degree = kernel, float(C), gamma, r, degree
        self.solver, self.class_weight, self.sv_tol, self.ridge = solver, class_weight, sv_tol, ridge

    def _K(self, A, B):
        return kernel_matrix(A, B, self.kernel, self.gamma_, self.r, self.degree)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        if set(np.unique(y)) <= {0.0, 1.0}:                 # accept {0,1} labels
            y = np.where(y == 0, -1.0, 1.0)
        N = len(y)
        self.gamma_ = 1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" else float(self.gamma)

        Ci = np.full(N, self.C)                             # per-class box constraint
        if self.class_weight == "balanced":
            Ci *= np.where(y == 1, N / (2 * (y == 1).sum()), N / (2 * (y == -1).sum()))

        t0 = time.perf_counter()
        K = self._K(X, X)
        Q = np.outer(y, y) * K + self.ridge * np.eye(N)

        # Note: the sigmoid kernel is not positive semidefinite, so Q can have negative
        # eigenvalues and the QP would be non-convex. Clip them to zero (nearest PSD matrix).
        self.psd_fixed_ = False
        if self.kernel == "sigmoid":
            w, V = np.linalg.eigh(Q)
            if w.min() < 0:
                Q = (V * np.clip(w, 0, None)) @ V.T + self.ridge * np.eye(N)
                self.psd_fixed_ = True

        a = self._cvxopt(Q, y, Ci) if self.solver == "cvxopt" else self._slsqp(Q, y, Ci)
        self.fit_time_ = time.perf_counter() - t0

        a = np.clip(a, 0, Ci)
        self.objective_ = 0.5 * a @ Q @ a - a.sum()
        tol = self.sv_tol * Ci.max()
        sv = a > tol
        free = sv & (a < Ci - tol)                          # 0 < alpha < C: on the margin
        self.alphas_ = a
        self.sv_X_, self.sv_y_, self.sv_alpha_ = X[sv], y[sv], a[sv]
        self.n_support_ = int(sv.sum())
        use = free if free.any() else sv
        self.b_ = float(np.mean(y[use] - K[use][:, sv] @ (a[sv] * y[sv])))
        self.w_ = (self.sv_alpha_ * self.sv_y_) @ self.sv_X_ if self.kernel == "linear" else None
        return self

    def _cvxopt(self, Q, y, Ci):
        N = len(y)
        i = np.arange(N)
        G = spmatrix(np.r_[-np.ones(N), np.ones(N)], np.r_[i, i + N], np.r_[i, i], (2 * N, N))
        h = matrix(np.r_[np.zeros(N), Ci])
        sol = solvers.qp(matrix(Q), matrix(-np.ones(N)), G, h,
                         matrix(y.reshape(1, -1)), matrix(0.0), options={"show_progress": False})
        self.status_ = sol["status"]
        return np.ravel(sol["x"])

    def _slsqp(self, Q, y, Ci):
        N = len(y)
        res = minimize(lambda a: 0.5 * a @ Q @ a - a.sum(), np.zeros(N), jac=lambda a: Q @ a - 1,
                       method="SLSQP", bounds=[(0, c) for c in Ci],
                       constraints=[{"type": "eq", "fun": lambda a: a @ y, "jac": lambda a: y}],
                       options={"maxiter": 500, "ftol": 1e-10})
        self.status_ = "optimal" if res.success else res.message
        return res.x

    def decision_function(self, X):
        return self._K(np.asarray(X, float), self.sv_X_) @ (self.sv_alpha_ * self.sv_y_) + self.b_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)

    def score(self, X, y):
        y = np.asarray(y, float).ravel()
        if set(np.unique(y)) <= {0.0, 1.0}:
            y = np.where(y == 0, -1.0, 1.0)
        return float(np.mean(self.predict(X) == y))


# ------------------------------------------------------------------ data
KERNELS = [("linear", {}), ("poly", dict(r=1.0, degree=3)),
           ("rbf", {}), ("sigmoid", dict(r=0.0))]


def synthetic_sets(n=300, seed=0):
    from sklearn.datasets import make_blobs, make_circles, make_moons
    pm1 = lambda y: np.where(y == 0, -1.0, 1.0)
    rng = np.random.default_rng(seed)
    Xx = rng.uniform(-1, 1, (n, 2))
    yx = np.where(Xx[:, 0] * Xx[:, 1] > 0, 1.0, -1.0)
    Xx = Xx + rng.normal(0, 0.05, Xx.shape)
    return {
        "blobs_separable": (lambda d: (d[0], pm1(d[1])))(make_blobs(n, centers=[(-2, -2), (2, 2)], cluster_std=0.8, random_state=seed)),
        "blobs_overlap": (lambda d: (d[0], pm1(d[1])))(make_blobs(n, centers=[(-1, -1), (1, 1)], cluster_std=1.3, random_state=seed)),
        "circles": (lambda d: (d[0], pm1(d[1])))(make_circles(n, noise=0.1, factor=0.4, random_state=seed)),
        "moons": (lambda d: (d[0], pm1(d[1])))(make_moons(n, noise=0.2, random_state=seed)),
        "xor": (Xx, yx),
    }


def split_scale(X, y, test_size=0.25, seed=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)                          # fit on train only (no leakage)
    return sc.transform(Xtr), sc.transform(Xte), ytr, yte


def plot_boundary(m, X, y, ax, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 250),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 250))
    Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9], colors=["#fbd5d5", "#d5e0fb"], alpha=.7)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=1)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", s=12)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="red", s=12)
    ax.scatter(m.sv_X_[:, 0], m.sv_X_[:, 1], s=70, facecolors="none", edgecolors="k")
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


# ------------------------------------------------------------------ experiments
def exp_toy():
    print("\n=== 1. Toy problem with known answer ===")
    X = np.array([[.5, 1], [.5, -1], [-.5, 0], [2, 2], [3, -1], [-2, 1], [-3, -2]])
    y = np.array([1, 1, -1, 1, 1, -1, -1], float)
    m = DualSVM("linear", C=10).fit(X, y)
    print("alphas:", np.round(m.alphas_, 5), " expected [1 1 2 0 0 0 0]")
    print("w, b  :", np.round(m.w_, 5), round(m.b_, 5), " expected [2 0], 0")


def exp_sklearn(out):
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.svm import SVC
    print("\n=== 2. Check against sklearn SVC (breast cancer) ===")
    d = load_breast_cancer()
    Xtr, Xte, ytr, yte = split_scale(d.data, np.where(d.target == 0, -1.0, 1.0))
    g = 1.0 / (Xtr.shape[1] * Xtr.var())
    rows = []
    for k, p in KERNELS:
        m = DualSVM(k, C=1.0, gamma=g, **p).fit(Xtr, ytr)
        s = SVC(kernel=k, C=1.0, gamma=g, coef0=p.get("r", 0), degree=p.get("degree", 3)).fit(Xtr, ytr)
        if k == "poly":      # sklearn poly = (gamma x.x' + coef0)^d, so use gamma=1 to match ours
            s = SVC(kernel="poly", C=1.0, gamma=1.0, coef0=p["r"], degree=p["degree"]).fit(Xtr, ytr)
        rows.append(dict(kernel=k, my_acc=m.score(Xte, yte), sk_acc=s.score(Xte, yte),
                         my_nSV=m.n_support_, sk_nSV=int(s.n_support_.sum())))
    df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
    df.to_csv(f"{out}/sklearn_check.csv", index=False)


def exp_synthetic(out):
    import matplotlib.pyplot as plt
    import pandas as pd
    print("\n=== 3. Synthetic datasets x 4 kernels ===")
    sets = synthetic_sets()
    fig, axes = plt.subplots(len(sets), 4, figsize=(13, 3.0 * len(sets)))
    rows = []
    for i, (name, (X, y)) in enumerate(sets.items()):
        Xtr, Xte, ytr, yte = split_scale(X, y)
        for j, (k, p) in enumerate(KERNELS):
            m = DualSVM(k, C=1.0, **p).fit(Xtr, ytr)
            rows.append(dict(dataset=name, kernel=k, train_acc=m.score(Xtr, ytr),
                             test_acc=m.score(Xte, yte), n_SV=m.n_support_, time_s=m.fit_time_))
            plot_boundary(m, Xtr, ytr, axes[i, j], f"{name} | {k} | test {rows[-1]['test_acc']:.2f}")
    plt.tight_layout(); plt.savefig(f"{out}/synthetic_boundaries.png", dpi=120); plt.close()
    df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
    df.to_csv(f"{out}/synthetic_results.csv", index=False)


def exp_sweeps(out):
    import matplotlib.pyplot as plt
    import pandas as pd
    print("\n=== 4. Sweeps on moons ===")
    X, y = synthetic_sets(400)["moons"]
    Xtr, Xte, ytr, yte = split_scale(X, y)
    rows = []
    def add(sweep, v, m):
        rows.append(dict(sweep=sweep, value=v, train_acc=m.score(Xtr, ytr),
                         test_acc=m.score(Xte, yte), n_SV=m.n_support_))
    for C in np.logspace(-2, 2, 9): add("C (rbf)", C, DualSVM("rbf", C=C).fit(Xtr, ytr))
    for g in np.logspace(-2, 2, 9): add("gamma (rbf)", g, DualSVM("rbf", gamma=g).fit(Xtr, ytr))
    for d in range(1, 9): add("degree (poly)", d, DualSVM("poly", degree=d).fit(Xtr, ytr))
    for g in np.logspace(-2, 1, 7): add("gamma (sigmoid)", g, DualSVM("sigmoid", gamma=g, r=0.0).fit(Xtr, ytr))
    df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
    df.to_csv(f"{out}/sweeps.csv", index=False)
    names = list(dict.fromkeys(df.sweep))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, s in zip(axes, names):
        d = df[df.sweep == s]
        ax.plot(d.value, d.train_acc, "o-", label="train"); ax.plot(d.value, d.test_acc, "s-", label="test")
        if "degree" not in s: ax.set_xscale("log")
        ax.set_title(s); ax.set_ylabel("accuracy"); ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(f"{out}/sweeps.png", dpi=120); plt.close()


def exp_optimizers(out, sizes=(100, 200, 300)):
    import pandas as pd
    print("\n=== 5. cvxopt vs SLSQP (RBF) ===")
    rows = []
    for n in sizes:
        X, y = synthetic_sets(n)["moons"]
        X = (X - X.mean(0)) / X.std(0)
        for s in ("cvxopt", "slsqp"):
            m = DualSVM("rbf", solver=s).fit(X, y)
            rows.append(dict(n=n, solver=s, time_s=m.fit_time_, dual_objective=m.objective_,
                             n_SV=m.n_support_, train_acc=m.score(X, y)))
    df = pd.DataFrame(rows); print(df.round(5).to_string(index=False))
    df.to_csv(f"{out}/optimizer_comparison.csv", index=False)


def exp_scaling(out, sizes=(100, 200, 400, 800, 1200)):
    import matplotlib.pyplot as plt
    import pandas as pd
    print("\n=== 6. Time vs n (cvxopt, RBF) ===")
    rows = []
    for n in sizes:
        X, y = synthetic_sets(n)["moons"]
        m = DualSVM("rbf").fit(X, y)
        rows.append(dict(n=n, time_s=m.fit_time_)); print(rows[-1])
    df = pd.DataFrame(rows); df.to_csv(f"{out}/scaling.csv", index=False)
    plt.figure(figsize=(5, 3.5)); plt.loglog(df.n, df.time_s, "o-"); plt.grid(alpha=.3, which="both")
    plt.xlabel("training size n"); plt.ylabel("fit time (s)"); plt.tight_layout()
    plt.savefig(f"{out}/scaling.png", dpi=120); plt.close()


def exp_csv(path, label, out, n_neg=1500):
    """Optional Kaggle CSV (e.g. creditcard.csv): all 4 kernels, with/without balanced class weights."""
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n=== 7. {path} ===")
    df = pd.read_csv(path)
    hi = df[label].max()
    df = pd.concat([df[df[label] == hi], df[df[label] != hi].sample(n=n_neg, random_state=42)]).sample(frac=1, random_state=42)
    X, y = df.drop(columns=[label]).values.astype(float), np.where(df[label].values == hi, 1.0, -1.0)
    Xtr, Xte, ytr, yte = split_scale(X, y, 0.2)
    for k, p in KERNELS:
        for cw in (None, "balanced"):
            m = DualSVM(k, C=1.0, class_weight=cw, **p).fit(Xtr, ytr)
            print(f"\n--- {k}, class_weight={cw}, nSV={m.n_support_}, time={m.fit_time_:.2f}s")
            pr = m.predict(Xte)
            print(classification_report(yte, pr, target_names=["neg(-1)", "pos(+1)"], digits=3))
            print(confusion_matrix(yte, pr))


def run_all(out="results", csv=None, label="Class"):
    import matplotlib
    matplotlib.use("Agg")
    os.makedirs(out, exist_ok=True)
    np.random.seed(0)
    exp_toy(); exp_sklearn(out); exp_synthetic(out); exp_sweeps(out)
    exp_optimizers(out); exp_scaling(out)
    if csv:
        exp_csv(csv, label, out)
    print(f"\nDone. CSVs and figures are in ./{out}/")
    try:                                                    # show figures inline in Colab/Jupyter
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import Image, display
            for f in ("synthetic_boundaries.png", "sweeps.png", "scaling.png"):
                display(Image(filename=os.path.join(out, f)))
    except Exception:
        pass


if __name__ == "__main__":
    run_all()
