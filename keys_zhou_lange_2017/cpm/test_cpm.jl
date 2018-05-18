n = 1000
M = randn(n,n)
M = 0.5*(M+M')

# Hall and Newman example
# n = 5
# M = ones(n,n)
# for j = 2:n
#    M[j,j-1] = -1.0
#    M[j-1,j] = -1.0
# end
# M[1,5] = -1.0
# M[5,1] = -1.0
# for i = 1:n
#    println(M[i,:])
# end

tic(); (iters,x) = copositivity(M); toc()
loss = 0.5*dot(x,M*x)
println("iters = ",iters," loss = ",loss," norm = ",norm(x)," min = ",minimum(x))
tic(); (iters,y) = accelerated_copositivity(M); toc()
loss = 0.5*dot(y,M*y)
println("iters = ",iters," loss = ",loss," norm = ",norm(y)," min = ",minimum(y))
# tic(); (iters,z) = accelerated_copositivity2(M); toc()
# loss = 0.5*dot(z,M*z)
# println("iters = ",iters," loss = ",loss," norm = ",norm(z)," min = ",minimum(z))
# println(" norm of difference = ",norm(x-y))
