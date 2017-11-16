# load libraries 
library(PMA)
library(Matrix)
library(compiler)

# precompile a function to round a number to a correct number of digits
prc = cmpfun(function(x, k) format(round(x, k), nsmall=k))

compare.spca = cmpfun(function(){

    # get dataset
    data(breastdata)

    # RNA data must be transposed before running
    X = t(breastdata$rna)
    p = dim(X)[2]
    x.means = apply(X, 2, mean)
    x = X
    for (i in 1:p){ x[,i] = x[,i] - x.means[i] }

    # set algo parameters
    sumabsv = 8
    niter   = 1000
    trace   = F
    orth    = T
    center  = F 
    vpos    = F
    vneg    = F
    cmp.pve = F
    K       = 25 
    v       = NULL
    seed    = 2016
    totvar  = sum(diag(t(x) %*% x))
    kvar0   = 0

    # header for output
    cat("#PC\tNnz\tObj\td(ortho)\tTime\tPVE\tAPVE\tCPVE\n")

    # set seed and then run algo once for each model size
    # ensure warm start each time
    set.seed(seed)
    for ( i in 1:K ){

        # time SPC 
        start.time = proc.time()
        my.spc = SPC(x, sumabsv=sumabsv, niter=niter, trace=trace, orth=orth, center=center, vpos=vpos, vneg=vneg, K=i, v=v, compute.pve = cmp.pve)
        stop.time = proc.time() - start.time

        # v is matrix of sparse loading vectors 
        v = as.matrix(my.spc$v)

        # xk is matrix of PCs 
        xk = x %*% v %*% solve( t(v) %*% v, t(v))

        # compute various variances
        kvar = sum(diag(t(xk) %*% xk))
        adjvar = kvar - kvar0
        dortho = sqrt(sum(t(v) %*% v - diag(i))^2)
        normvar = sum(diag(t(v) %*% t(x) %*% x %*% v))

        # print output
        cat(paste(i, "\t", nnzero(v[,i]), "\t", round(normvar), "\t", prc(dortho, 3), "\t", prc(stop.time[3], 3), "\t", round(kvar), round(adjvar), prc(kvar / totvar,2), "\n"))

        # save previous variance of k PCs
        kvar0 = kvar
    }
#    return(my.spc)
     return()
})

#my.spc = compare.spca()
compare.spca()
