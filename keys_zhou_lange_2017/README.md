### Proximal Distance Algorithms: Theory and Practice

This folder contains code for the manuscript by KL Keys, H Zhou, K Lange. 
All proximal distance algorithm code is in Julia.
The subfolders are

* `lp` for linear programming
* `nqp` for nonnegative quadratic programming
* `socp` for second order cone programming
* `spca` for sparse principal components analysis
* `spm` for sparse precision matrix

## External software

The paper uses external software packages for comparison of algorithmic performance.
To correctly reproduce the comparisons requires installation of the software.
It may also require valid commercial licenses.

# R 

The sparse PCA example uses the R package [`PMA`](https://cran.r-project.org/web/packages/PMA/index.html) for comparison.
Users can download it from [CRAN](https://cran.r-project.org/web/packages/PMA/index.html) or from within R directly via the command

    install.packages("PMA")

# Gurobi and MOSEK
Several comparisons rely on the [Gurobi](http://www.gurobi.com/) or [MOSEK](https://www.mosek.com/) solvers.
These are high-quality commercial optimization solvers.
The paper used a free Gurobi academic license, available [here](http://www.gurobi.com/registration/academic-license-reg), and a free MOSEK academic license, available [here](https://www.mosek.com/products/academic-licenses/).

# SCS 
The [Splitting Cone Solver](https://github.com/cvxgrp/scs) (SCS) is an open-source cone solver that uses the ADMM framework.
It is freely available on Github [here](https://github.com/cvxgrp/scs). 

# Ipopt

The [Interior Point Optimizer](https://projects.coin-or.org/Ipopt) is an open-source nonlinear solver.
The paper used Ipopt via [Ipopt.jl](https://github.com/JuliaOpt/Ipopt.jl).
Installing Ipopt.jl also installs the solver.
