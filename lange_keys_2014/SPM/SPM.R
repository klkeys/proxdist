#!/usr/bin/R

# shore up memory
rm(list = ls())

# set random seed
set.seed(12345);

# loss function
f = function(X,S,egs){
	return(- sum(log(egs)) + sum(diag(S%*%X)));
}
library(glasso);
library(Matrix);
library(MASS);

N = 2^(3:9);
#N = 2^(3:8);
p = 1000;

# let's get super Bayesian!
# following list contains regularization paths that should recover true model in each dimension
# no thanks to GLASSO authors for making this easy :-P

paths = list(
	seq(0,0.003,0.0002),            # path for n = 8 
	seq(0.0030,0.0032,0.00002),     # path for n = 16
	seq(0.0032,0.0034,0.00001),     # n = 32
	seq(0.0044,0.0046,0.00001),     # 64
	seq(0.005,0.0051,0.000001),     # 128
	seq(0.00661,0.00664,0.0000002), # 256
	seq(0.0098,0.0099,0.000001)     # 512
);

for (i in 1:length(N)){ 
	
	# dimension of problem
	n = N[i];

	cat(paste("Dimension of precision matrix is ", n, " times ", n, ".\n", sep = ""));
	cat(paste("True model has ", 3*n - 6, " elements in bands.\n", sep = ""));

	# incrementer for eventually averaging over multiple runs
	# also, cap runs (here, at 10)
	multiruns = 1;
	max_runs = 10;

	# vectors to save information for SPM routine in MATLAB
	K = matrix(-1, max_runs, ncol = 1);
	band.K= matrix(-1, max_runs, ncol = 1);
	losses = matrix(-1, max_runs, 1);
	glasso.times = matrix(-1, max_runs, 1)	
	rhos.to.ks = matrix(-1, max_runs, 1)	

	while (multiruns <= max_runs){
		x = matrix(rnorm(n*p), nrow = n);
		U = band(matrix(rnorm(n*n),n,n),0,3);
		P = t(U) %*% U + 0.01 * x %*% t(x);
		P = (P + t(P)) / 2;
		s2 = as.matrix(solve(P));
		s2 = (s2 + t(s2)) / 2; # symmetrized covariance matrix

		# shore up memory
		rm(x,U,P);

		# calculate regularization path
#		glasso.time = proc.time();
		rhos = paths[[i]] 
		a = glassopath(s2, rholist = rhos, penalize.diagonal = 0, trace = 0, thr = 1e-6);
#		glasso.time = proc.time() - glasso.time;
		wee = a$wi; # precision matrices
		#cat(paste("GLASSO took ", glasso.time[3], " seconds.\n", sep = ""));
		
		# get information for precision matrices along regularization path
		# will quit when path has diagonal precision matrix
		# only save information for rho that corresponds to # of elements in bands
		for (j in 1:length(rhos)){
			
			# rho for corresponding precision matrix in "wee"
			rho = rhos[j];
			
			# precision matrix 
			Z = wee[,,j];
			
			# total nonzero entries in upper triangle?
			temp = abs(triu(Z, k = 1)); 
			rho.to.k = floor(sum(as.numeric(temp > 1e-6)));
			
			# stopping criterion; no need to continue along path once precision matrix is diagonal
			# occurs when rho translates to k = 0
			if(rho.to.k == 0){
				cat("\n");
				break;
			}

			# how many entries should be in band?
			total.k = (n - 1) + (n - 2) + (n - 3);

			# when total.k is same as rho.to.k, save output
			if(rho.to.k == total.k){

				#cat("Found rho that yields desired number of elements in true model!\n");
				#cat("Running glasso() with that value of rho...\n");
				glasso.time = proc.time();
				aa = glasso(s = s2, rho = rho, thr = 1e-6, penalize.diagonal = FALSE,
							start = "warm", wi.init = solve(s2), w.init = s2); 
				glasso.time = proc.time() - glasso.time;
				Z2 = aa$wi;
			
				# total nonzero entries in upper triangle?
				temp = abs(triu(Z2, k = 1)); 
				rho.to.k = floor(sum(as.numeric(temp > 1e-6)));

#				cat(paste("total.k = ", total.k, "rho.to.k = ", rho.to.k, "\n", sep = "")); 
				if(total.k == rho.to.k){
				
					# want to count true positives
					# count along first three (super/sub)diagonals
					k = 0;
					temp = abs(Z2);
					
					the.diag = diag(temp[,-1]);
					k = k + sum(as.numeric(the.diag > 1e-6)); 
					the.diag = diag(t(temp)[,-1]);
					k = k + sum(as.numeric(the.diag > 1e-6)); 

					the.diag = diag((temp[,-1])[,-1]);
					k = k + sum(as.numeric(the.diag > 1e-6)); 
					the.diag = diag((t(temp)[,-1])[,-1]);
					k = k + sum(as.numeric(the.diag > 1e-6)); 

					the.diag = diag(((temp[,-1])[,-1])[,-1]);
					k = k + sum(as.numeric(the.diag > 1e-6)); 
					the.diag = diag(((t(temp)[,-1])[,-1])[,-1]);
					k = k + sum(as.numeric(the.diag > 1e-6)); 

					# correct for double counting on k
					k = floor(k / 2);
					band.K[multiruns] = k;
							
					# save this rho as correlate to k
					K[multiruns] = rho.to.k;

					# calculate loss function for current precision matrix and write to file
					eigs = eigen(Z2, only.values = T);
					loss = f(Z,s2,eigs$values);
					losses[multiruns] = loss;

					# save compute time and corresponding rho
					glasso.times[multiruns] = glasso.time[3];
					rhos.to.ks[multiruns] = rho;
				
					# monitor algorithm progress	
					cat(paste("rho = ", rho, "\tcorresponding k = ", rho.to.k, ",\t# elements in bands = ", k, ",\tglasso loss = ", loss, ",\tglasso time = ", glasso.time[3], "\n", sep = ""));

					# save current precision, covariance matrices to file, clear from memory, and continue to next run
					write.table(format(Z, digits = 12), paste("./SPM/W_", n, "_", rho*10000000, "_", multiruns,".txt", sep = ""), row.names = F, col.names = F, quote = F);
					write.table(format(s2, digits = 12), paste("./SPM/X_", n, "_", multiruns, ".txt", sep = ""), row.names = F, col.names = F, quote = F);
					rm(Z,s2);

					# keep track of number of good runs so far!
					multiruns = multiruns + 1;
					break;
				}
				else{
					#cat("glasso() failed! Proceeding to next rho...\n");
				}
			}
		}
	}
	# write all collected information to file
	write.table(format(glasso.times, digits = 12), paste("./SPM/glassotime_", n, ".txt", sep = ""), row.names = F, col.names = F, quote = F);
	write.table(format(losses, digits = 12), paste("./SPM/losses_", n,".txt", sep = ""), row.names = F, col.names = F, quote = F);
	write.table(format(K, digits = 12), paste("./SPM/K_", n,".txt", sep = ""), row.names = F, col.names = F, quote = F);
	write.table(format(band.K, digits = 12), paste("./SPM/bandK_", n, ".txt", sep = ""), row.names = F, col.names = F, quote = F);
	write.table(format(rhos.to.ks, digits = 12), paste("./SPM/rhos_", n,".txt", sep = ""), row.names = F, col.names = F, quote = F);

	# shore up memory
	rm(glasso.time, losses, K, band.K, rhos, rhos.to.ks, temp, the.diag, wee, a, eigs, n, rho, rho.to.k, i, j, k);	
}
rm(list = ls());
