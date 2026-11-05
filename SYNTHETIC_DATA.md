# Synthetic data for the custom SVM

Run from the repository root (Python 3, no third-party dependencies):

```sh
python3 code/generate_synthetic.py
```

This creates eight separate binary classification problems in `data/synthetic/`.
Each dataset folder contains `train.csv` (300 points), `validation.csv` (100),
and `test.csv` (100). Columns are `x1,x2,y`, with labels `-1,+1`.
The features live in R²: the label is not a third input feature.
The generator uses NumPy neither for data creation nor for file loading.

Open `data/synthetic/preview.svg` in a browser to inspect the training points
and known generating boundaries. `metadata.json` records rules, seeds, split
sizes, and class counts. Default seed: 582. Re-running the same command
replaces these generated files with the same data.

| Dataset | Construction | Purpose |
| --- | --- | --- |
| linear | Uniform square, label by sign(x1+x2), omit a strip around the boundary | Clear linear separation and margins |
| linear_noisy | Same linear rule without the gap; flip labels independently with probability 10% | Soft margins and noise; realized flip rate varies |
| circles | Uniform square, positive inside radius sqrt(8/pi) | Nonlinear boundary; about half positive in expectation |
| xor | Positive when x1*x2 > 0 | Interaction features and nonlinear kernels |
| linear_imbalanced | Same clear-margin linear rule; about 10% positive in each split | Class weights, minority recall, and balanced accuracy |
| quadratic | Positive when x2 >= x1² - 0.5 | Parabolic boundary; degree-2 polynomial features |
| polynomial_degree4 | Positive when x2 >= 0.3x1⁴ - x1² + 0.5 | Degree-4 boundary; compare polynomial degrees and RBF |
| mixed_geometry | Positive above 0.5x1 on the left (x1 < 0), and above x1² - x1 on the right (x1 >= 0) | Continuous boundary combining linear and curved regions |

All cases have two input dimensions, including the degree-4 case. Polynomial
degree describes boundary complexity, not the number of input columns. The
mixed case is one classification task with a piecewise boundary; the model
does not receive a region label or select a separate kernel for each point.
Circles and XOR also have degree-2 polynomial decision rules, so they are not
exclusive tests of RBF kernels. A suitable kernel makes a boundary expressible;
finite-sample learning and parameter selection still determine performance.

Each split is sampled independently with its own deterministic random seed.
Cases other than linear_imbalanced have natural class counts, not forced equal
counts; those counts depend on the boundary geometry and the random sample.
The imbalanced case fixes its class counts using rejection sampling.
All coordinates are raw: any learned preprocessing must be fitted on training
data and then applied unchanged to validation and test data.

## Load into the existing model

From the repository root, this loads its class without executing `run_all()`:

```python
import runpy
import sys
sys.path.insert(0, "code")
from generate_synthetic import load_csv

DualSVM = runpy.run_path("code/test_svm", run_name="svm_module")["DualSVM"]
X_train, y_train = load_csv("data/synthetic/linear/train.csv")
X_val, y_val = load_csv("data/synthetic/linear/validation.csv")
X_test, y_test = load_csv("data/synthetic/linear/test.csv")

model = DualSVM(kernel="linear", C=1.0).fit(X_train, y_train)
print("Validation accuracy:", model.score(X_val, y_val))
# After choosing the kernel and parameters on validation data:
print("Test accuracy:", model.score(X_test, y_test))
```

The custom model requires NumPy, SciPy, and CVXOPT, independently of this
generator. This example does not call sklearn's SVM or the model file's
reference-comparison experiment.

Fit a new model for each problem. Try linear, polynomial (degrees 2, 3, and 4), and RBF
kernels; choose C and any kernel parameters with validation data. Keep the
test data out of parameter selection. For imbalanced data, include a confusion
matrix, recall for +1, and balanced accuracy rather than only accuracy.
Repeat with several seeds to assess variability:

```sh
python3 code/generate_synthetic.py --seed 583 --output data/synthetic_seed583
```

Synthetic success verifies behavior on these controlled examples; it does not
establish generalization to unrelated real datasets. Train separately on each
real dataset. The existing `code/main.py` has an unrelated syntax error and is
not used by these commands. The sigmoid implementation also needs review
before being used for comparisons.
