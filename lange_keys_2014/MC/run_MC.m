% run_MC
% RUN MATRIX COMPLETION ON MULTIPLE DIMENSIONS
%
% This script runs the matrix completion routine for various dimensions n. 
% The script also runs a MATLAB implementation of SoftImpute. It will save 
% all console output to the file MC_test.txt. Some of the code is taken
% from the example for SoftImpute. If used, please cite the reference
% 
% Mazumder R, Hastie T, Tibshirani R (2010) 
% Spectral regularization algorithms for learning large incomplete matrices. 
% J Machine Learning Res 11:2287?2322
%
% coded by Kevin L. Keys (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% clean slate!
clear;
clc;

% save console output
diary('MC_test.txt');

% timestamp for start of code
fprintf(['% Start time: ', datestr(now)]);
fprintf('\n');

fprintf('- - - - - - - - - - - - - - - - - -\n');
fprintf('- - - BEGIN MATRIX COMPLETION - - -\n');
fprintf('- - - - - - - - - - - - - - - - - -\n');

% fix random seeds for reproducibility
randn('state',2009);
rand('state',2009);

% roughly control # of cases to test, so actual entries of adjustments do
% not matter as much as length of it does. higher length corresponds to
% more tested cases, higher dimensions, higher ranks, etc.
adjustments = 1:10;


fprintf('$m$\t$n$\t$\\alpha$\ttrue rank\trank\t$L_1$         $L_2$         $L_2 - L_1$\t\t$\\lambda$\t$T_1$\t$T_2$\tDiff(RelPredError)\n');
path_length = 20;
output = zeros(length(adjustments) * path_length,12);
for j = 1:length(adjustments)

    nadj    = 5 * j;
    madj    = 4 * j;
    r_adj   = 2 * j;
    nrow    = 50  * madj; 
    ncol    = 50  * nadj; 
    rnk     = 10   * r_adj; 
    density = 0.05 * j; 

    R = sprand(nrow,ncol,density);
    [i_row,j_col,~] = find(R); 
    clear R*;

    %------- low-rank factorization & population outer-product
    Ueff    = randn(nrow,rnk);
    Veff    = randn(ncol,rnk); 
    X_clean = Ueff * Veff';  

    temp_vec  = Ueff(i_row,:);
    temp_vecr = Veff(j_col,:);
    TTemp     = dot(temp_vec,temp_vecr,2);
    
    % uncomment next lines to add noise to observed matrix
%     sig=0.05;
%     TTemp=TTemp+ sig*randn(length(TTemp),1);

    %------ sparse-observed matrix "GXobs" 
    GXobs = sparse(i_row,j_col,TTemp,nrow,ncol);
    GPm   = sparse(i_row,j_col,1,nrow,ncol);
    clear temp_vec temp_vecr TTemp;

    % declare fields of structure OPTS
    OPTS = [];
    OPTS.TOLERANCE   = 1e-6;
    OPTS.MAXITER     = 1e4; 
    OPTS.SMALL_SCALE = 0;
    
    % approximates the lambda value for which solution is zero
    lambda_max       = spectral_norm(GXobs); 

    %**************************************
    % create a path of solutions
    %************************************** 

    error      = zeros(1,path_length);
    error_MM   = error;
    lambda_seq = linspace(lambda_max*.9,lambda_max*.1,path_length);
    W          = sparse(i_row,j_col,ones(size(i_row)));

    INIT = [];
    for i = 1:path_length
        
        % want CPU time on SI algorithm
        tic;
        
        % warm-start specified via INIT
        [a11,a22,a33,~] = soft_impute(GXobs,lambda_seq(i),OPTS,INIT); 
        softimpute_time = toc;
        
        % reconstruct matrix from SVD output of SI
        Z = a11 * a22 * a33';
        
        % want CPU time on MM algorithm
        tic;
        [X,u,s,v] = MatrixCompletion(GXobs,GPm,nnz(a22),Z);  % warm start
%         [X,u,s,v] = MatrixCompletion(GXobs,GPm,nnz(a22));    % cold start
        mm_time = toc;

        % loss functions
        lossX = 0.5 * norm(GPm .* (GXobs - X), 'fro')^2;
        lossZ = 0.5 * norm(GPm .* (GXobs - Z), 'fro')^2;
        
        % relative prediction error
        error    = norm( (1 - GPm) .* (Z - X_clean), 'fro') ...
                   / norm( (1 - GPm) .* (X_clean), 'fro');
        error_MM = norm( (1 - GPm) .* (X - X_clean), 'fro') ...
                   / norm( (1 - GPm) .* (X_clean), 'fro');
        
        fprintf('%d\t%d\t%0.2f\t%d\t\t%d\t%3.6f\t%3.6f\t%3.6f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n', ...
            nrow,ncol,density,rnk,nnz(a22),lossX,lossZ,lossZ-lossX,lambda_seq(i),mm_time,softimpute_time,error-error_MM);

        % specify warm-starts for next (smaller) lambda value
        INIT = struct('U', a11 , 'D', a22,  'V', a33 ); 

        % save output
        output(j*i,:) = [nrow,ncol,density,rnk,nnz(a22),lossX,lossZ,...
        lossZ-lossX,lambda_seq(i),mm_time,softimpute_time,error-error_MM];

        
        % free some memory
        clear a11 a22 a33 X Z u s v error error_MM lossX lossZ mm_time softimpute_time;
    end
    fprintf('\n');
end

clear INIT OPTS GXobs GPm i_row j_col X_clean nadj madj r_adj nrow ncol ...
 rnk density adjustments i j path_length Ueff Veff W lambda_max lambda_seq;

% bookkeeping
fprintf(['% End time: ', datestr(now)]);
fprintf('\n');
diary off;
save(['MC_',datestr(now, 'ddmmmyyyy')]);

if 0 % comment to test some specific cases
% ms = [1200, 2000, 5000];
% ns = [1500, 2500, 5000];
% densities = [0.15, 0.10, 0.05];
% rnks = [40, 20, 30];
ms = 5000; ns = 5000; densities = 0.05; rnks = 30;
% path_length = 3;
path_length = 5;
% specific_cases = zeros(length(ms) * path_length,12);

for j = 1:length(ms)

    nrow    = ms(j); 
    ncol    = ns(j); 
    rnk     = rnks(j); 
    density = densities(j); 

    R = sprand(nrow,ncol,density);
    [i_row,j_col,~] = find(R); 
    clear R*;

    %------- low-rank factorization & population outer-product
    Ueff    = randn(nrow,rnk);
    Veff    = randn(ncol,rnk); 
    X_clean = Ueff * Veff';  

    temp_vec  = Ueff(i_row,:);
    temp_vecr = Veff(j_col,:);
    TTemp     = dot(temp_vec,temp_vecr,2);
    
    % uncomment next lines to add noise to observed matrix
%     sig=0.05;
%     TTemp=TTemp+ sig*randn(length(TTemp),1);

    %------ sparse-observed matrix "GXobs" 
    GXobs = sparse(i_row,j_col,TTemp,nrow,ncol);
    GPm   = sparse(i_row,j_col,1,nrow,ncol);
    clear temp_vec temp_vecr TTemp;

    % declare fields of structure OPTS
    OPTS = [];
    OPTS.TOLERANCE   = 1e-6;
    OPTS.MAXITER     = 1e4; 
    OPTS.SMALL_SCALE = 0;
    
    % approximates the lambda value for which solution is zero
    lambda_max       = spectral_norm(GXobs); 

    %**************************************
    % create a path of solutions
    %************************************** 

    error      = zeros(1,path_length);
    error_MM   = error;
    lambda_seq = linspace(lambda_max*.9,lambda_max*.1,path_length);
    W          = sparse(i_row,j_col,ones(size(i_row)));

    INIT = [];
    for i = 1:path_length
        
        % want CPU time on SI algorithm
        tic;
        
        % warm-start specified via INIT
        [a11,a22,a33,~] = soft_impute(GXobs,lambda_seq(i),OPTS,INIT); 
        softimpute_time = toc;
        
        % reconstruct matrix from SVD output of SI
        Z = a11 * a22 * a33';
        
        % want CPU time on MM algorithm
        tic;
        [X,u,s,v] = MatrixCompletion(GXobs,GPm,nnz(a22),Z);  % warm start
%         [X,u,s,v] = MatrixCompletion(GXobs,GPm,nnz(a22));    % cold start
        mm_time = toc;

        % loss functions
        lossX = 0.5 * norm(GPm .* (GXobs - X), 'fro')^2;
        lossZ = 0.5 * norm(GPm .* (GXobs - Z), 'fro')^2;
        
        % relative prediction error
        error    = norm( (1 - GPm) .* (Z - X_clean), 'fro') ...
                   / norm( (1 - GPm) .* (X_clean), 'fro');
        error_MM = norm( (1 - GPm) .* (X - X_clean), 'fro') ...
                   / norm( (1 - GPm) .* (X_clean), 'fro');
        
        fprintf('%d\t%d\t%0.2f\t%d\t\t%d\t%3.6f\t%3.6f\t%3.6f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n', ...
            nrow,ncol,density,rnk,nnz(a22),lossX,lossZ,lossZ-lossX,lambda_seq(i),mm_time,softimpute_time,error-error_MM);

        % specify warm-starts for next (smaller) lambda value
        INIT = struct('U', a11 , 'D', a22,  'V', a33 ); 

        % save output
%         specific_cases(j*i,:) = [nrow,ncol,density,rnk,nnz(a22),lossX,lossZ,...
%         lossZ-lossX,lambda_seq(i),mm_time,softimpute_time,error-error_MM];

        
        % free some memory
        clear a11 a22 a33 X Z u s v error error_MM lossX lossZ mm_time softimpute_time;
    end
    fprintf('\n');
end

% clear memory
clear INIT OPTS GXobs GPm i_row j_col X_clean ms ns rnks densities nrow ncol ...
 rnk density adjustments i j path_length Ueff Veff W lambda_max lambda_seq;
end
