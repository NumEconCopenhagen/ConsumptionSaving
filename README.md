# ConSav: Consumption-Saving Models

A code library for solving and simulating consumption-saving models in Python using Numba JIT compiled functions. The **consav** package provides:

* A **Model** class with predefined methods for e.g. saving and loading derived from the [EconModel package](https://github.com/NumEconCopenhagen/EconModel)
* A **multi-linear interpolation** module
* Optimizers such as **golden section search** and **newton-raphson**
* An **upper envelope** function for using the endogenous grid point method in non-convex models

All of the above is written to be **Numba compatible**.

The repository **[ConsumptionSavingNotebooks](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks)** contains a number of examples on using the various tools and two models:

* The canonical buffer-stock consumption model
* A durable consumption models with non-convex adjustment costs

The library builds on the model class in **[EconModel](https://github.com/NumEconCopenhagen/EconModel)**. The repository **[EconModelNotebooks](https://github.com/NumEconCopenhagen/EconModelNotebooks)** contains a number of examples on using this underlying model class.

**New to Python?** Try out this online course, [Introduction to programming and numerical analysis](https://numeconcopenhagen.netlify.com/).

**Interested in other numerical tools for economists in Python?** Try out [QuantEcon](https://lectures.quantecon.org/).

# Installation

The package can be installed with

```
pip install EconModel
pip install ConSav
```

**Requirements:** 

1. `python >= 3.8`
2. `numba >= 0.52`

# Development

To develop the package follow these steps:

1. Clone this repository
2. Locate the cloned repostiory in a terminal
4. Run `pip install -e .`

Changes you make to the package is now immediately effective on your own computer. 
