import numpy as np
import kagglehub

def get_data():
    pass_fail_path = kagglehub.dataset_download("ishanjha100/student-passfail-data")
    cancer_path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
    credit_path = kagglehub.dataset_download("rohitudageri/credit-card-details")
    loan_path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
    airplane_path = kagglehub.dataset_download("threnjen/2019-airline-delays-and-cancellations")
    bank_path = kagglehub.dataset_download("ritesaluja/bank-note-authentication-uci-data")




# Kernels
def linear_kernel(X1, X2):
    return X1 @ X2.T

def polynomial_kernel(X1, X2, degree=2, coef=1):
    return (X1 @ X2.T + coef) ** degree

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



def main()

    pass

main()