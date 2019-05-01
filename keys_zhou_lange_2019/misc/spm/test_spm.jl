using ProxOpt
using Distances
include("spm.jl")

function test_spm()

    # set random seed
    seed = 2016
    srand(seed)

    # function parameters
    max_iter = 10000
    rho      = 1.0 
    rho_inc  = 1.2 
    inc_step = 1
    rho_max  = 1e30
    tol      = 1e-6
    stol     = 1e-6
    quiet    = true

    # what is core file path to SPM files from glasso?
    filepath = "/Users/kkeys/Desktop/proxdist/spm/"

    # LaTeX formatted header
    println("\\begin{table}")
    println("\t\\centering")
    println("\t\\begin{tabular}{cccccccc}")
    println("\t\t\\multicolumn{2}{c}{Dimensions} & \\multicolumn{3}{c}{Optima} & \\multicolumn{3}{c}{CPU Times} \\\\")
    println("\t\t\\cmidrule(r){1-2} \\cmidrule(r){3-5} \\cmidrule(r){6-8} \\\\")
    println("\t\t\$p\$ & \$k_{t}\$ & \$k_{1}\$ & \$k_{2}\$ & \$\\rho\$ & \$L_1\$ & \$L_2\$ & \$T_{1}\$ & \$T_{2}\$ \\\\") 
    println("\t\t\\hline")

    # set parameters for simulation
    N     = collect(3:9)
    nruns = 10

    # loop through files
    for m = 1:length(N)

        # dimension of current problem
        n = 2^N[m] 
        full_k = n - 1 + n - 2 + n - 3
        
        # file names for current dimension
        Kfilename      = filepath * "K_$n.txt"
        gl_Kfilename   = filepath * "bandK_$n.txt"
        lossesfilename = filepath * "losses_$n.txt"
        rhofilename    = filepath * "rhos_$n.txt"
        gltimefilename = filepath * "glassotime_$n.txt"

        # load files
        K      = vec(readdlm(Kfilename, Int))
        gl_K   = vec(readdlm(gl_Kfilename))
        losses = vec(readdlm(lossesfilename))
        rhos   = vec(readdlm(rhofilename))
        gltime = vec(readdlm(gltimefilename))
        
        tot_gl_loss = 0
        tot_rho     = 0
        tot_gl_k    = 0
        tot_mm_k    = 0
        tot_loss    = 0
        tot_mmtime  = 0
        tot_gltime  = 0
        

        # SPM.R should produce nruns per dimension n, so iterate over each
        # run and accumulate information for each dimension. we will average
        # everything at the end
        for r = 1:nruns
            
            # read covariance matrix from file
            Xfilename = filepath * "X_$(n)_$(r).txt"
            s = readdlm(Xfilename)
#            st = s'
            
            # function handle for loss, coded with current covariance matrix s
#            f(x) = -logdet(x) + vecdot(st,x)
            
            # read variables from GLASSO for this run
#            k       = K[r]
            k       = full_k
            gl_k    = gl_K[r]
            rho     = rhos[r]
            gl_loss = losses[r]
            gl_time = gltime[r]

            # filler matrix for warm start
            # can start it very close to empirical precision matrix
            Y = inv(s);

            # run SPM, record compute time and true positives
            tic()
            output = spm(s,k, rho=rho, rho_inc=rho_inc, rho_max=rho_max, tol=tol, sparsetol=stol, max_iter=max_iter, inc_step=inc_step, quiet=quiet, Y=Y)
            W_stoptime = toq()
            temp = spzeros(n,n)
            for d in (-3,-2,-1,1,2,3)
                temp += spdiagm(diag(output["X"],d), d, n, n)
            end
            mm_k = floor(countnz(temp) / 2)

            # accumulate variables
            tot_gl_loss += gl_loss
            tot_rho     += rho
            tot_gl_k    += gl_k
            tot_mm_k    += mm_k
            tot_loss    += output["obj"] 
            tot_mmtime  += W_stoptime
            tot_gltime  += gl_time
        end
        
        # print output for this dimension
        @printf("\t\t\$%d\$ & \$%d\$ & \$%3.1f\$ & \$%3.1f\$ & \$%7.7f\$ & \$%3.2f\$ & \$%3.2f\$ & \$%3.3f\$ & \$%3.3f\$ \\\\ \n", n, full_k, tot_mm_k / nruns, tot_gl_k / nruns, tot_rho / nruns, tot_loss / nruns, tot_gl_loss / nruns, tot_mmtime / nruns, tot_mmtime / tot_gltime)
    end

    println("\t\t\\bottomrule")
    println("\t\\end{tabular}")
    println("\t\\caption{Numerical results for precision matrix estimation. Abbreviations: \$p\$  for the matrix dimension, \$k_{t}\$ for the number of nonzero entries in the true model, \$k_{1}\$ for the number of true nonzero entries recovered by the proximal distance algorithm, \$k_{2}\$ for the number of true nonzero entries recovered by \\texttt{glasso}, \$\\rho\$ the average \\texttt{glasso} tuning constant for a given \$k_{t}\$, \$L_{1}\$ the average loss from the proximal distance algorithm, \$L_{2}\$ the average loss from \\texttt{glasso}, \$T_{1}\$ the average compute time in seconds for the proximal distance algorithm, and \$T_{1} / T_{2}\$ the ratio of \$T_{1}\$ to the average compute time for \\texttt{glasso}.} \\\\") 
    println("\t\\label{tab:spm}")
    println("\\end{table}")

    return nothing
end

test_spm()
