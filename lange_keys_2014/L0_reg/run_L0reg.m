% run_L0reg
% RUN L0 REGRESSION ON MULTIPLE DIMENSIONS
%
% This script runs the L0 sparse regression routine for various problem 
% dimensions n. The script also runs a MATLAB implementation of LASSO. 
% It will save all console output to the file L0_reg_test.txt. 
%
% coded by Kevin L. Keys (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

clear;
clc;

diary('L0_reg_test.txt')

% timestamp for start of code
fprintf(['Start time: ', datestr(now)]);
fprintf('\n');

fprintf('- - - - - - - - - - - - - - - - - -\n');
fprintf('- - -   BEGIN L0 REGRESSION   - - -\n');
fprintf('- - - - - - - - - - - - - - - - - -\n');
fprintf('\n\n');

% fix random seeds for reproducibility
randn('state',2014);
rand('state',2014);

% how many different scenarios should we test?
numcases = 5;

% a macro for computing the residual sum of squares
lossfunc = @(b,x,y) sum( (x*b - y).^2 );

% these two matrices store output
% output2 stores the transposed case of output
output  = zeros(numcases,10);
output2 = zeros(numcases,10);

% header for output, formatted for LaTeX table
fprintf('$m$ & $n$ & $df$ & $tp_{1}$ & $tp_{2}$ & $\\lambda$ & $L_1$ & $L_1 / L_2$ & $T_1$ & $T_1 / T_2$ \\\\ \n');

% what are the nonzero values of our tested beta vector?
beta_nonzeroes = [1,1/2,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/10];

% how many nonzeroes do we have?
beta_df = length(beta_nonzeroes);
for j = 1:numcases

    % try large and sparse examples
    % will flip m, n later and rerun
    m = pow2(7 + j);
    n = pow2(6 + j);

    lasso_truepos     = 0;
    mm_truepos        = 0;
    mm_totalloss      = 0;
    lasso_totalloss   = 0;
    m_totaltime       = 0;
    mm_totaltime2     = 0;
    lasso_losscounter = 0;
    lambda_ave        = 0;
    
	% run 100 instances of the problem dimensions
    for dummy = 1:100

        % initialize data and model 
        X    = randn(m,n);
        beta = zeros(n,1);

		% fill model with nonzero values
        beta(1:beta_df) = beta_nonzeroes;

		% create noisy response
        Y = X*beta + randn(m,1);

        % solve lasso regularized least squares
        % order of arguments is
        % -- design matrix X
        % -- response vector Y
        tic;
        [betas, stats] = lasso(X,Y, 'DFmax', beta_df);

        % stop timer
        m_time = toc;
        m_totaltime = m_totaltime + m_time;

        % will calculate total MM time over "path" of dfs
        mm_totaltime = 0;
        mm_path      = zeros(n,beta_df);
        x_mm         = randn(n,1);

		% loop over number of nonzeroes in beta
		% this computes the "regularization path" for L0	
        for k = 1:beta_df
            tic;
            [x_mm, ~, ~, ~] = L0_reg(X,Y,k,x_mm);
            mm_time         = toc;
            mm_totaltime    = mm_totaltime + mm_time;
            mm_loss         = lossfunc(x_mm,X,Y);
            mm_path(:,k)    = x_mm;

			% if we hit the end of the path, then compare to lasso
            if k == beta_df
%				x_save         = x_mm;

				% save the betas corresponding to the current number of nonzeroes 
				% save the corresponding lambda values as well
                comp_save      = betas(:,find(stats.DF == beta_df));
                lambdas        = stats.Lambda(find(stats.DF == beta_df));

				% compute the loss function for all betas from lasso
				% we want the index corresponding to the minimum loss
				% save the value of the RSS and the lambda for that index
                loss_funcs     = bsxfun(@minus, X*comp_save, Y);
                losses         = sum(loss_funcs .^ 2, 1);
                [lasso_loss,i] = min(losses);
                lambda         = lambdas(i);

				% if lasso_loss is not empty, then save the results
                if numel(lasso_loss)
                    mm_totalloss      = mm_totalloss + mm_loss;
                    lasso_totalloss   = lasso_totalloss + lasso_loss;
                    lasso_losscounter = lasso_losscounter + 1;
                    lambda_ave        = lambda_ave + lambda;
                end
            end
        end
        mm_totaltime2 = mm_totaltime2 + mm_totaltime;

        true_mm_nonzeroes = zeros(1,beta_df); 
        for a = 1:beta_df
            true_mm_nonzeroes(1,a) = nnz(mm_path(1:beta_df,a)); 
        end
        betas = fliplr(betas);
        true_lasso_nonzeroes = zeros(1,size(betas,2)); 
        for a = 1:size(betas,2) 
            true_lasso_nonzeroes(1,a) = nnz(betas(1:beta_df,a)); 
        end

        ltp = max(true_lasso_nonzeroes(find(fliplr(stats.DF) == beta_df)));
        mmtp = true_mm_nonzeroes(beta_df);
        
        % guard against empty arrays
        if ~numel(ltp)
            ltp = 0;
        end
        if ~numel(mmtp)
            mmtp = 0;
        end
        
%         disp([betas(1:beta_df,find(fliplr(stats.DF) == beta_df)), ones(beta_df,1), mm_path(1:10,beta_df)]);        
        lasso_truepos = lasso_truepos + ltp;
        mm_truepos = mm_truepos + mmtp;
    end
        lambda_ave      = lambda_ave / lasso_losscounter;
        lasso_truepos   = lasso_truepos / dummy;
        mm_truepos      = mm_truepos / dummy;
        lasso_totalloss = lasso_totalloss / lasso_losscounter;
        mm_totalloss    = mm_totalloss / lasso_losscounter;
        mm_totaltime2   = mm_totaltime2 / dummy;
        m_totaltime     = m_totaltime / dummy;
        
%         fprintf('%d & %d & %d & %3.2f & %3.2f & %3.3f & %3.3f & %3.3f & %3.3f & %3.3f \\\\ \n', ...
%             m, n, beta_df, mm_truepos, lasso_truepos, lambda, ...
%             mm_loss, mm_loss / lasso_loss, mm_totaltime, mm_totaltime / m_time);
% 
%         output(j,:) = [m, n, beta_df, mm_truepos, lasso_truepos, lambda, ...
%             mm_loss, mm_loss / lasso_loss, mm_totaltime, mm_totaltime / m_time];

        fprintf('%d & %d & %d & %3.2f & %3.2f & %3.3f & %3.3f & %3.3f & %3.3f & %3.3f \\\\ \n', ...
            m, n, beta_df, mm_truepos, lasso_truepos, lambda_ave, ...
            mm_totalloss, mm_totalloss / lasso_totalloss, mm_totaltime2, mm_totaltime2 / m_totaltime);

        output(j,:) = [m, n, beta_df, mm_truepos, lasso_truepos, lambda_ave, ...
            mm_totalloss, mm_totalloss / lasso_totalloss, mm_totaltime2, mm_totaltime2 / m_totaltime];

        % shore up memory
        clear m n mm_truepos lasso_truepos lambda mm_loss lasso_loss ...
            mm_totaltime m_time true_lasso_nonzeroes true_mm_nonzeroes ...
            k i x_save comp_save lambdas loss_funcs losses ...
            mm_totaltime mm_time x_mm beta Y m n lambda_ave ...
            m_totaltime mm_totaltime2 mmtp ltp a;
        clear betas mm_path stats;
    
    
    % flip m, n later and rerun
    n = pow2(7 + j);
    m = pow2(6 + j);

    lasso_truepos     = 0;
    mm_truepos        = 0;
    mm_totalloss      = 0;
    lasso_totalloss   = 0;
    m_totaltime       = 0;
    mm_totaltime2     = 0;
    lasso_losscounter = 0;
    lambda_ave        = 0;
    
    for dummy = 1:100
        % initialize model
        X = randn(m,n);
        beta = zeros(n,1);
        beta(1:beta_df) = beta_nonzeroes;
        Y = X*beta + randn(m,1);

        % solve lasso regularized least squares
        % order of arguments is
        % -- design matrix X
        % -- response vector Y
        tic;
        [betas, stats] = lasso(X,Y, 'DFmax', beta_df);

        % stop timer
        m_time = toc;
        m_totaltime = m_totaltime + m_time;

        % will calculate total MM time over "path" of dfs
        mm_totaltime = 0;

        mm_path = zeros(n,beta_df);
        x_mm = randn(n,1);
        for k = 1:beta_df
            tic;
            [x_mm, ~, ~, ~] = L0_reg(X,Y,k,x_mm);
            mm_time = toc;
            mm_totaltime = mm_totaltime + mm_time;
            mm_loss = lossfunc(x_mm,X,Y);
            mm_path(:,k) = x_mm;
            if k == beta_df
    %             x_save = x_mm;
                comp_save = betas(:,find(stats.DF == beta_df));
                lambdas = stats.Lambda(find(stats.DF == beta_df));
                loss_funcs = bsxfun(@minus, X*comp_save, Y);
                losses = sum(loss_funcs .^ 2, 1);
                [lasso_loss,i] = min(losses);
                lambda = lambdas(i);
                if numel(lasso_loss)
                    mm_totalloss = mm_totalloss + mm_loss;
                    lasso_totalloss = lasso_totalloss + lasso_loss;
                    lasso_losscounter = lasso_losscounter + 1;
                    lambda_ave = lambda_ave + lambda;
                end
            end
        end
        mm_totaltime2 = mm_totaltime2 + mm_totaltime;

        true_mm_nonzeroes = zeros(1,beta_df); 
        for a = 1:beta_df
            true_mm_nonzeroes(1,a) = nnz(mm_path(1:beta_df,a)); 
        end
        betas = fliplr(betas);
        true_lasso_nonzeroes = zeros(1,size(betas,2)); 
        for a = 1:size(betas,2) 
            true_lasso_nonzeroes(1,a) = nnz(betas(1:beta_df,a)); 
        end

        ltp = max(true_lasso_nonzeroes(find(fliplr(stats.DF) == beta_df)));
        mmtp = true_mm_nonzeroes(beta_df);
        
        % guard against empty arrays
        if ~numel(ltp)
            ltp = 0;
        end
        if ~numel(mmtp)
            mmtp = 0;
        end
        
%         disp([betas(1:beta_df,find(fliplr(stats.DF) == beta_df)), ones(beta_df,1), mm_path(1:10,beta_df)]);        
        lasso_truepos = lasso_truepos + ltp;
        mm_truepos = mm_truepos + mmtp;
    end
    lambda_ave      = lambda_ave / lasso_losscounter;
    lasso_truepos   = lasso_truepos / dummy;
    mm_truepos      = mm_truepos / dummy;
    lasso_totalloss = lasso_totalloss / lasso_losscounter;
    mm_totalloss    = mm_totalloss / lasso_losscounter;
    mm_totaltime2   = mm_totaltime2 / dummy;
    m_totaltime     = m_totaltime / dummy;

%         fprintf('%d & %d & %d & %3.2f & %3.2f & %3.3f & %3.3f & %3.3f & %3.3f & %3.3f \\\\ \n', ...
%             m, n, beta_df, mm_truepos, lasso_truepos, lambda, ...
%             mm_loss, mm_loss / lasso_loss, mm_totaltime, mm_totaltime / m_time);
% 
%         output(j,:) = [m, n, beta_df, mm_truepos, lasso_truepos, lambda, ...
%             mm_loss, mm_loss / lasso_loss, mm_totaltime, mm_totaltime / m_time];

    fprintf('%d & %d & %d & %3.2f & %3.2f & %3.3f & %3.3f & %3.3f & %3.3f & %3.3f \\\\ \n', ...
        m, n, beta_df, mm_truepos, lasso_truepos, lambda_ave, ...
        mm_totalloss, mm_totalloss / lasso_totalloss, mm_totaltime2, mm_totaltime2 / m_totaltime);

    output2(j,:) = [m, n, beta_df, mm_truepos, lasso_truepos, lambda_ave, ...
        mm_totalloss, mm_totalloss / lasso_totalloss, mm_totaltime2, mm_totaltime2 / m_totaltime];

    % shore up memory
    clear m n mm_truepos lasso_truepos lambda mm_loss lasso_loss ...
        mm_totaltime m_time true_lasso_nonzeroes true_mm_nonzeroes ...
        k i x_save comp_save lambdas loss_funcs losses ...
        mm_totaltime mm_time x_mm beta Y m n lambda_ave ...
        m_totaltime mm_totaltime2 mmtp ltp a lasso_losscounter ...
        lasso_totalloss mm_totalloss;
    clear betas mm_path stats;
end

output = [output; output2];

clear output2 numcases lossfunc i j m n beta_df dummy X;

% bookkeeping
fprintf(['\n\nEnd time: ', datestr(now)]);
fprintf('\n');
diary off;
save(['L0_reg_',datestr(now, 'ddmmmyyyy')]);
% end
