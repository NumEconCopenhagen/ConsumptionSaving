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

**New to Python?:** Why out this online course, [Introduction to programming and numerical analysis](https://numeconcopenhagen.netlify.com/).
