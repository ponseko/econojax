# EconoJax: A Fast & Scalable Economic Simulation in JAX

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

EconoJax is (loosely) a reimplementation of [The AI Economist](https://www.science.org/doi/10.1126/sciadv.abk2607) in JAX with a 1D observation space rather than the original 2D visual space.
With GPU support, EconoJax's transition function is over 100x times faster and agents converge over **2000x** times faster.

---

## 📦 Installation

For those using [uv](https://docs.astral.sh/uv/getting-started/installation/), it is possible to run a standard PPO implementation with default settings by directly running `uv run main.py`.

```bash
git clone git@github.com:ponseko/econojax.git
cd econojax
uv run main.py
```

Alternatively, install the project as an editable package in your favourite virtual environment software. E.g. using conda:

```bash
git clone git@github.com:ponseko/econojax.git
cd econojax
conda create -n econojax python=3.11
conda activate econojax
pip install -e .

python main.py
```

for CUDA support, additionally run `pip install jax[cuda]`.

---

## 📑 Citing

If you use EconoJax in your research or projects, please cite:

```bibtex
@article{ponse2024econojax,
  title={EconoJax: A Fast \& Scalable Economic Simulation in Jax},
  author={Ponse, Koen and Plaat, Aske and van Stein, Niki and Moerland, Thomas M},
  journal={arXiv preprint arXiv:2410.22165},
  year={2024}
}
```
