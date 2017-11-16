% run_SPM
% RUN SPARSE PRECISION MATRIX INFERENCE ON MULTIPLE DIMENSIONS
%
% This script runs the SPM routine for various dimensions n. It depends
% intimately on previous output from the SPM.R script, coded in R. The
% screen output is saved to SPM.txt in a LaTeX-formatted table. Note that
% the program will run into problems if it cannot find the output from
% SPM.R. Set the file handles to the appropriate paths.
%
% coded by Kevin L. Keys (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

clear;
clc;
% system('R CMD BATCH /Users/kkeys/Desktop/Seoul/SPM.R');

% record all screen output
diary on;
diary('SPM.txt');

% timestamp for start of code
fprintf(['% Start time: ', datestr(now)]);
fprintf('\n');

% set random seed
rng(12345);
fprintf('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n');
fprintf('-             Running precision matrix program!             -\n'); 
fprintf('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n');


% LaTeX formatted header
fprintf('$d$ & $k_{t}$ & $k_{1}$ & $k_{2}$ & $\\rho$ & $L_1$ & $L_2 - L_1$ & $T_1$ & $T_1 / T_2$ \\\\ \n');
fprintf('\\hline \n');
N = pow2(3:9);
for m = 1:length(N);
% for m = 1:1
    n = N(m);
    full_k = (n - 1 + n - 2 + n - 3);
    
    Kfilename = ['/Users/kkeys/Desktop/Seoul/SPM/K_', num2str(n),'.txt'];
    K = dlmread(Kfilename);
    gl_Kfilename = ['/Users/kkeys/Desktop/Seoul/SPM/bandK_', num2str(n),'.txt'];
    gl_K = dlmread(gl_Kfilename);
    lossesfilename = ['/Users/kkeys/Desktop/Seoul/SPM/losses_', num2str(n),'.txt'];
    losses = dlmread(lossesfilename);
    rhofilename = ['/Users/kkeys/Desktop/Seoul/SPM/rhos_', num2str(n),'.txt'];
    rhos = dlmread(rhofilename);
    glassotimefilename = ['/Users/kkeys/Desktop/Seoul/SPM/glassotime_', num2str(n),'.txt'];
    glassotime = dlmread(glassotimefilename);
    
    tot_glasso_loss = 0;
    tot_rho = 0;
    tot_gl_k = 0;
    tot_mm_k = 0;
    tot_loss = 0;
    tot_mmtime = 0;
    tot_glassotime = 0;
    

    % SPM.R should produce ten runs per dimension n, so iterate over each
    % run and accumulate information for each dimension. we will average
    % everything at the end
    for run = 1:10
        
        % read covariance matrix from file
        Xfilename = ['/Users/kkeys/Desktop/Seoul/SPM/X_', num2str(n), ...
            '_', num2str(run), '.txt'];
        s = dlmread(Xfilename);
        
        % function handle for loss, coded with current covariance matrix s
        f = @(x,y) -sum(log(x)) + trace(s * y);
        
        % read variables from GLASSO for this run
        k = K(run);
        gl_k = gl_K(run);
        rho = rhos(run);
        glasso_loss = losses(run);
        glasso_time = glassotime(run);

        % filler matrix for warm start
        % can start it very close to empirical precision matrix
        W = inv(s);

        % run SPM, record compute time and true positives
        tic;
        [X,d] = SparsePrecisionMatrix(s,uint32(k),W);
        W_stoptime = toc;
        mm_k = floor(nnz(spdiags(X, [-3:-1,1:3])) / 2);
        loss = f(d, X);

        % accumulate variables
        tot_glasso_loss = tot_glasso_loss + glasso_loss;
        tot_rho = tot_rho + rho;
        tot_gl_k = tot_gl_k + gl_k;
        tot_mm_k = tot_mm_k + mm_k;
        tot_loss = tot_loss + loss;
        tot_mmtime = tot_mmtime + W_stoptime;
        tot_glassotime = tot_glassotime + glasso_time;

    end
    
    % print output for this dimension
    fprintf('$%d$ & $%d$ & $%3.1f$ & $%3.1f$ & $%7.7f$ & $%3.2f$ & $%3.2f$ & $%3.3f$ & $%3.3f$ \\\\ \n', ...
            n, full_k, tot_mm_k / (run), tot_gl_k / (run), ...
            tot_rho / (run), tot_loss / (run), ...
            (tot_glasso_loss - tot_loss) / (run), tot_mmtime / (run), ...
            tot_mmtime / tot_glassotime);
end

% bookkeeping
fprintf(['% End time: ', datestr(now)]);
fprintf('\n');
diary off;
save(['SPM_',datestr(now, 'ddmmmyyyy')]);