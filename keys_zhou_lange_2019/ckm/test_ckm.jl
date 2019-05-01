include("ckm.jl")

function test_ckm()

    srand(1237)
    iterations = zeros(Int, 4)
    loss       = zeros(4)
    exec_time  = zeros(4)

    # formatted LaTeX table header
    println("\\begin{table}")
    println("\t\\centering")
    println("\t\\begin{tabular}{ccccccccc}")
    println("\t\t\t\\toprule")
    println("\t\t\t\\multicolumn{1}{c}{Size} & \\multicolumn{2}{c}{PD1} & \\multicolumn{2}{c}{PD2} & \\multicolumn{2}{c}{PD3} & \\multicolumn{2}{c}{Dykstra} \\\\")
    println("\t\t\t\\cmidrule(r){1-1} \\cmidrule(r){2-3} \\cmidrule(r){4-5} \\cmidrule(r){6-7} \\cmidrule(3){8-9}")
    println("\t\t\t\$n\$ & Loss & Time & Loss & Time & Loss & Time \\\\")
    println("\t\t\t\\hline")

    for i = 1:8

        # set problem size
        n = 2^i

        # random initial matrix
        M = randn(n,n)

        # symmetrize the matrix
        M = (M + M') / 2

        # first run: vanilla proxdist
        tic()
        (iters,Y) = closest_kinship_matrix(M)
        t = toq()
        #println(n,"  ",iters)
        iterations[1] = iters
        loss[1] = 0.5*vecnorm(Y-M)^2
        exec_time[1] = t

        # second run: accelerated proxdist
        tic()
        (iters,Y) = accelerated_closest_kinship_matrix(M)
        t = toq()
        #println(n,"  ",iters)
        iterations[2] = iters
        loss[2] = 0.5*vecnorm(Y-M)^2
        exec_time[2] = t

        # third run: accelerated proxdist with PSD domain constraints
        t = tic()
        (iters,Y) = accelerated_closest_kinship_matrix2(M)
        t = toq()
        #println(n,"  ",iters)
        iterations[3] = iters
        loss[3] = 0.5*vecnorm(Y-M)^2
        exec_time[3] = t

        # fourth run: Dykstra
        tic()
        (iters,Y) = dykstra_closest_kinship_matrix(M)
        t = toq()
        #println(n,"  ",iters)
        iterations[4] = iters
        loss[4] = 0.5*vecnorm(Y-M)^2
        exec_time[4] = t

        # print a line of the table
        @printf("\t\t\t %d & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f & %3.2f \\\\\n", n, loss[1], exec_time[1], loss[2], exec_time[2], loss[3], exec_time[3], loss[4], exec_time[4])

    end

    # print table footer
    println("\t\t\t\\bottomrule")
    println("\t\t\\end{tabular}")
    println("\t\\caption{CPU times and optima for the closest kinship matrix problem. Here the kinship matrix is \$n \\times n\$, PD1 is the proximal distance algorithm, PD2 is the accelerated proximal distance, PD3 is the accelerated proximal distance algorithm with the positive semidefinite constraints folded into the domain of the loss, and Dykstra is Dykstra's adaptation of alternating projections. All times are in seconds.}")
    println("\t\\label{tab:kin}")
    println("\\end{table}")

    return
end

test_ckm()
