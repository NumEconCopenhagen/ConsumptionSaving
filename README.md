# ConSav: Consumption-Saving Models

A code library for solving and simulating consumption-saving models in Python using Numba JIT compiled functions. The **consav** package provides:

* A **ConsumptionSavingModel** class with predefined methods for e.g. saving and loading
* A **multi-linear interpolation** module
* Optimizers such as **golden section search** and **newton-raphson**
* An **upper envelope** function for using the endogenous grid point method in non-convex models
* Functions for interfacing easily with c++

All of the above is written to be **Numba compatible**.

The repository **[ConsumptionSavingNotebooks](https://github.com/NumEconCopenhagen/ConsumptionSavingNotebooks)** contains a number of examples on using the various tools and two models:

* The canonical buffer-stock consumption model
* A durable consumption models with non-convex adjustment costs

_The library is still early in its development, contributions are very welcome!_

**New to Python?** Try out this online course, [Introduction to programming and numerical analysis](https://numeconcopenhagen.netlify.com/).

**Interested in other numerical tools for economists in Python?** Try out [QuantEcon](https://lectures.quantecon.org/).

# Installation

The package can be installed with

```
pip install git+https://github.com/NumEconCopenhagen/ConsumptionSaving
```

# Development

To develop the package follow these steps:

1. Install watchdog, `pip install watchdog`
2. Clone this repository
3. Locate the cloned repostiory in a terminal
4. Run `python setyp.py watch`

This re-installs the package whenever you make any changes. 
