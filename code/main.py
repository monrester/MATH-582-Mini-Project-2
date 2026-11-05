import numpy as np
from pathlib import Path



def get_data():
    data_path = Path(__file__).resolve().parent.parent / "data"
    bank = np.loadtxt(data_path / "bank.csv", delimiter=",", skiprows=1)
    cancer = np.loadtxt(data_path / "cancer.csv", delimiter=",", skiprows=1)
    loan = np.loadtxt(data_path / "loan.csv", delimiter=",", skiprows=1)
    pass_fail = np.loadtxt(data_path / "pass-fail.csv", delimiter=",", skiprows=1)

    datasets = {"bank": bank,
                "cancer": cancer,
                "loan": loan,
                "pass_fail": pass_fail}
    return datasets





# Kernels

#No parameters for linear kernel
def linear_kernel(X1, X2):
    return X1 @ X2.T

#Degree and coef are the two parameters
def polynomial_kernel(X1, X2, degree=2, coef=1):
    return (X1 @ X2.T + coef) ** degree

#Gamma is the only parameter
def rbf_kernel(X1, X2, gamma=1):
    sq1 = np.sum(X1 **2, axis=1).reshape(-1,1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(-1,1)
    sq_dists = sq1 + sq2 - 2 *(X1 @ X2.T)
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-gamma * sq_dists)


#check if the kernel is symmetric and posi semi-definite
def valid_kernel(K):
    symmetric = np.allclose(K, K.T)
    smallest_eig = np.linalg.eigvalsh(K).min()
    return symmetric and smallest_eig > -(1e-8)



def main():
    data = get_data()
    print(data["pass_fail"])

main()