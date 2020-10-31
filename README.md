# Including partial differential equations in your PyMC3 model
### Ivan Yashchuk
### PyMCon 2020 Tutorial on using FEniCS with PyMC3

This tutorial intoduces the use of [FEniCS](http://fenicsproject.org/) for solving differentiable variational problems in [PyMC3](https://docs.pymc.io/).

Automatic adjoint solvers for FEniCS programs are generated with [dolfin-adjoint/pyadjoint](http://www.dolfin-adjoint.org/en/latest/). These solvers make it possible to use Theano's (PyMC3 backend) reverse mode automatic differentiation with FEniCS.

Here is the link to the static (non-interactive) version of the tutorial: https://ivanyashchuk.github.io/fenics_pymccon2020/

## Setup

This tutorial assumes that you have [Anaconda](https://www.anaconda.com/distribution/#download-section) (Python 3.7 version) setup and installed on your system.

The next step is to clone or download the tutorial materials in this repository. If you are familiar with Git, run the clone command:

    git clone https://github.com/IvanYashchuk/fenics_pymccon2020.git
    
otherwise you can [download a zip file](https://github.com/IvanYashchuk/fenics_pymccon2020/archive/main.zip) of its contents, and unzip it on your computer.

The repository for this tutorial contains a file called `environment.yml` that includes a list of all the packages used for the tutorial. If you run:

    conda env create -f environment.yml
    
from the main tutorial directory, it will create the environment for you and install all of the packages listed. This environment can be enabled using:

    conda activate fenics_pymc3_tutorial
    
Then, I recommend using JupyterLab to access the materials (`tutorial.ipynb`):

    jupyter lab
