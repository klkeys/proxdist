source("ckm.jl")

funcion test_ckm()

    srand(1237)
    outfile    = "Kin.out"
    io         = open(outfile, "w")
    iterations = zeros(Int, 4)
    loss       = zeros(4)
    exec_time  = zeros(4)
    #
    for i = 1:8
      n = 2^i
      M = randn(n,n)
      M = 0.5*(M + M')
    #
      tic()
      (iters,Y) = closest_kinship_matrix(M)
      t = toc()
      println(n,"  ",iters)
      iterations[1] = iters
      loss[1] = 0.5*vecnorm(Y-M)^2
      exec_time[1] = t
    #
      tic()
      (iters,Y) = accelerated_closest_kinship_matrix(M)
      t = toc()
      println(n,"  ",iters)
      iterations[2] = iters
      loss[2] = 0.5*vecnorm(Y-M)^2
      exec_time[2] = t
    #
      t = tic()
      (iters,Y) = accelerated_closest_kinship_matrix2(M)
      t = toc()
      println(n,"  ",iters)
      iterations[3] = iters
      loss[3] = 0.5*vecnorm(Y-M)^2
      exec_time[3] = t
    #
      tic()
      (iters,Y) = dykstra_closest_kinship_matrix(M)
      t = toc()
      println(n,"  ",iters)
      iterations[4] = iters
      loss[4] = 0.5*vecnorm(Y-M)^2
      exec_time[4] = t
    #
      println(io,n," & ",round(loss[1], 2)," & ",round(exec_time[1], 2)," & ",
        round(loss[2], 2)," & ",round(exec_time[2], 2)," & ",
        round(loss[3], 2)," & ",round(exec_time[3], 2)," & ",
        round(loss[4], 2)," & ",round(exec_time[4], 2))
    end
    close(io)

    return()
end

test_ckm()
