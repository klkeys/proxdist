using RCall

function compare_spca(proj_type::ASCIIString = "column")

    # set seed
    seed = 2016
    srand(2016)

    tol     = 1e-4    # convergence tolerance
    feastol = 1e-3    # admissible distance to feasibility at convergence 
    quiet   = true    # turn on output? use this for debugging

    # comparison parameters
    R = vec([130 133 126 122 162 178 183 172 138 174 193 165 160 403 173 173 166 217 222 185 148 148 196 156 198])
    k = length(R)     # number of PCs, in simulation we use k = 25

    # need to get breast cancer RNA data from PMA package
    # matrix is annoyingly stored in transpose, so untranspose it
    x = rcopy("library(PMA); data(breastdata); t(breastdata\$rna)")
    n,p = size(x)

    # must center x
    for i = 1:p
        x[:,i] = (x[:,i] - mean(x[:,i])) 
    end

    # now load our SPCA functions
    include("spca.jl")

    # need SVD for warm start
    u,s,v = svd(x)

    # total variance in dataset?
    totvar = trace(x'*x)

    # track previous variance of k-1 components
    kvar0 = 0.0

    # name the "output" variable in order to return SPCA results
    output = 0

    # precompile SPCA by running once and discarding results
    V = v[:,1:1]
    output = spca(x, 1, R[1], proj_type, U=V, quiet=quiet, feastol=feastol, tol=tol) 

    # spca() uses accelerated PD algo with ortho domain constraints
    println("PCs\tNnz\tObj\tdortho\tTime\tVE\tAVE\tPVE\tIter")
    for i = 1:k 

        # V is warm start for SPCA variable U
        V = v[:,1:i]

        # compute SPCA
        tic()
        output = spca(x, i, R[1:i], proj_type, U=V, quiet=quiet, feastol=feastol, tol=tol) 
        mm_time = toq()

        # get sparse loadings
        u = copy(output["U"])

        # xk is matrix of k PCs
        xk = x * u * ((u'*u) \ u')

        # compute various variances
        kvar    = trace(xk'*xk)
        adjvar  = kvar - kvar0
        dortho  = vecnorm(u'*u - I)
        normvar = trace(u'*x'*x*u)

        # print output
        if proj_type == "matrix"
            @printf("%d\t%d\t%3.0f\t%3.3f\t%3.3f\t%3.0f\t%3.0f\t%3.3f\t%d\n", i, countnz(u), normvar, dortho, mm_time, kvar, adjvar, kvar / totvar, output["iter"])
        else
            @printf("%d\t%d\t%3.0f\t%3.3f\t%3.3f\t%3.0f\t%3.0f\t%3.3f\t%d\n", i, countnz(u[:,i]), normvar, dortho, mm_time, kvar, adjvar, kvar / totvar, output["iter"])
        end

        # reset V, kvar0
        fill!(V, 0.0)
        kvar0 = kvar
    end 

    return nothing
end

println("# Results from matrix projection")
compare_spca("matrix")

println("# Results from columnwise projection")
compare_spca()

println("# Remember: added compute cost of SVD warm start in Julia is ~2 sec!")
