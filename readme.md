---

# FirstOrderOptim

A modular Python framework for experimenting with first-order optimization algorithms on models and test functions. This project is designed for educational purposes. It only relies on 'numpy' for computation, and 'matplotlib' for visualization. The goal is to provide a clear understanding of how each optimization method works, with clean, readable code.


📚 A detailed blog post will come soon.

---

## Features

- **Linear Regression** (dense and sparse examples)
- **Classical Test Functions** (Booth, Three-Hump Camel)
- **Optimizers:** SGD, Momentum, Nesterov Momentum, AdaGrad (more to come)
- **Custom Training Loop** with batching and shuffling
- **Pre-Implemented Experiments** via a launcher script
- **Modular code structure**

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/FirstOrderOptim.git
cd FirstOrderOptim
pip install numpy matplotlib
```

---

## Usage

### Run an Experiment

Use the launcher script to select and run an experiment:

```bash
python3 main.py linear_dense
python3 main.py linear_sparse
python3 main.py test_function
```

This will internally run the corresponding script in the experiments folder using Python’s module system.

---

## Project Structure

```
FirstOrderOptim/
│
├── main.py                      # Launcher script
├── experiments/
│   ├── linear_dense.py          # Dense linear regression experiment
│   ├── linear_sparse.py         # Sparse linear regression experiment
│   └── test_function.py         # Test function optimization experiment
├── models/
│   ├── linear_model.py
│   └── test_functions.py
├── optimizers/
│   ├── SGD.py
│   ├── momentum.py
│   ├── NesterovMomentum.py
│   └── adagrad.py
├── losses/
│   └── mse.py
├── utils/
│   └── trainer.py
```

---

## Customization

- **Add new optimizers:** Create a new class in optimizers inheriting from `BaseOptimizer`.
- **Add new models or test functions:** Extend models with your own classes.
- **Change training parameters:** Edit the relevant script in experiments.
