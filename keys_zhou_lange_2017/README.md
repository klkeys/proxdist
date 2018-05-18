# Proximal Distance Algorithms: Theory and Practice

This folder contains code for the proximal distance algorithm manuscript by KL Keys, H Zhou, and K Lange. 
All examples are coded in [Julia](https://julialang.org/). 
The subfolders correspond to the numerical examples from the manuscript: 

* `cpm` for copositive matrix
* `lc` for linear complementarity
* `lp` for linear programming
* `nqp` for nonnegative quadratic programming
* `socp` for second order cone programming
* `spca` for sparse principal components analysis
* `spm` for sparse precision matrix

## External software

The paper uses some external software packages for algorithmic performance comparisons.
To correctly reproduce the comparisons requires installation of the software.
It may also require valid commercial licenses.

### R 

The sparse PCA example uses the R package [`PMA`](https://cran.r-project.org/web/packages/PMA/index.html) for comparison.
Users can download it from [CRAN](https://cran.r-project.org/web/packages/PMA/index.html) or from within R directly via the command

    install.packages("PMA")

### Gurobi and MOSEK

Several comparisons rely on the [Gurobi](http://www.gurobi.com/) or [MOSEK](https://www.mosek.com/) solvers.
These are high-quality commercial optimization solvers.
The paper used a free Gurobi academic license, available [here](http://www.gurobi.com/registration/academic-license-reg), and a free MOSEK academic license, available [here](https://www.mosek.com/products/academic-licenses/).

### SCS 

The [Splitting Cone Solver](https://github.com/cvxgrp/scs) (SCS) is an open-source cone solver that uses the ADMM framework.
It is freely available on Github [here](https://github.com/cvxgrp/scs). 

### Ipopt

The [Interior Point Optimizer](https://projects.coin-or.org/Ipopt) is an open-source nonlinear solver.
The paper used Ipopt via [Ipopt.jl](https://github.com/JuliaOpt/Ipopt.jl).
Installing Ipopt.jl also installs the solver.
