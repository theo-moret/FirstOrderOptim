# First-Order-Optim  

A minimal, **NumPy-only** playground for first-order optimisation algorithms  
(SGD, Momentum, Nesterov, AdaGrad … and more on the way).

The code is deliberately small and self-contained so you can read every line,
tweak the maths, and run quick experiments in a notebook.

---

## Quick install 

```bash
git clone https://github.com/theo-moret/FirstOrderOptim
cd FirstOrderOptim
python -m venv .venv && source .venv/bin/activate
pip install -e .  
```

*Requires Python ≥ 3.12.*



## Ready-made experiments 

Each experiment is a normal Python module, so you can launch them with  
`python -m …` or the small `main.py` helper:

```bash
# dense linear regression
python -m first_order_optim.experiments.linear_dense
# or
python main.py linear_dense
```

Available experiments:

| Command          | What it does                          |
|------------------|---------------------------------------|
| `linear_dense`   | usual linear regression task          |
| `linear_sparse`  | sparse data, for AdaGrad              |
| `test_function`  | 2-D toy surface: ThreeHumpCamel       |

---

## Project layout 

```
FirstOrderOptim/
├── pyproject.toml               ← single source of build metadata
├── main.py                      ← tiny launcher 
└── src/
    └── first_order_optim/       ← import package root
        ├── experiments/         ← runnable examples
        ├── loss/                ← MSE, ...
        ├── model/               ← LinearModel, ThreeHumpCamel, ...
        ├── optimizer/           ← SGD, Momentum, AdaGrad, ...
        ├── scheduler/           ← DecayRateScheduler, ...
        └── utils/               ← Trainer, ...
```

Thanks to the **src-layout**, you can open a notebook anywhere and
still `import first_order_optim …` after the editable install.

---

## License

MIT — do whatever you like.
