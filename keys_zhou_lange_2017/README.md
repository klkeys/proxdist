# Proximal Distance Algorithms: Theory and Practice

This folder contains code for the manuscript by KL Keys, H Zhou, K Lange. 
All proximal distance algorithm code is in Julia.
The subfolders are

* `lp` for *l*inear *p*rogramming
* `nqp` for *n*nonnegative *q*uadratic *p*rogramming
* `socp` for *s*econd *o*rder *c*one *p*rogramming
* `spca` for *s*parse *p*rincipal *c*omponents *a*nalysis
* `spm` for *s*parse *p*recision *m*atrix

## External software

The paper uses external software packages for comparison.
To correctly reproduce the comparisons requires installation of the software.
It may also require valid commercial licenses.

# R 

The sparse PCA example uses the R package [`PMA`](https://cran.r-project.org/web/packages/PMA/index.html) for comparison.
Users can download it from CRAN from within R via the command

    install.packages("PMA")

# Gurobi and MOSEK
Several comparisons rely on the [Gurobi](http://www.gurobi.com/) or [MOSEK](https://www.mosek.com/) solvers.
These are high-quality commercial optimization solvers.
The paper used a free Gurobi academic license, available [here](http://www.gurobi.com/registration/academic-license-reg), and a free MOSEK academic license, available [here](https://www.mosek.com/products/academic-licenses/).

# SCS 
The [Splitting Cone Solver](https://github.com/cvxgrp/scs) (SCS) is an open-source cone solver that uses the ADMM framework.
It is freely available on Github [here](https://github.com/cvxgrp/scs). 

# Ipopt

The Interior Point Optimizer is an open-source nonlinear solver.
The paper used Ipopt via [Ipopt.jl](https://github.com/JuliaOpt/Ipopt.jl).
Installing Ipopt.jl also installs the solver.
