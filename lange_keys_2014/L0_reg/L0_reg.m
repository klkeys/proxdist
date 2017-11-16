function [x_mm, MM_iter, MM_time, MM_opt] = L0_reg(X, Y, k, varargin)
% LINEAR REGRESSION WITH AN L0 PENALTY
% This function solves L0-penalized least squares
%
%   minimize || XB - Y ||_2^2 + w || B ||_0
%   
% by exact penalization and proximal mapping. It uses an MM proximal
% mapping algorithm to solve the problem. To simplify the notation, which 
% uses various forms of X, we rewrite the problem as
%
% minimize 0.5 * x'Ax + b'x + w || x - P(x) ||_2^2
%
% where A = X'X, b = -X'Y, x = B, and P projects onto the k components of x
% with largest magnitude.
%
% Arguments:
%
% -- X is the design matrix.
% -- Y is the response vector.
% -- k is the desired model size.
%
% Coded by Kevin L. Keys (2014)
% klkeys@g.ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    % global parameters
    tolerance   = 1e-6;
    max_iter    = 1e4;
    epsilon_min = 1e-15;
    rho_inc     = 1.2;
    eps_dec     = 1.2;
    [m,n]       = size(X);
    
    % initialize all output variables to zero.
    MM_iter     = 0;
    MM_time     = 0;
    MM_opt      = 0;

    % check for required variables
    if ~exist('X', 'var') || isempty(X)
        error('Need design matrix X.');
    end
    if ~exist('Y', 'var') || isempty(Y)
        error('Need response vector Y.');
    end
    if ~exist('k', 'var') || isempty(k)
        error('Need sparsity constraint k.');
    end
    
    % check for warm-start
    if nargin > 3
        x_mm = varargin{1};
    else
        if issparse(X)
            x_mm = sprandn(n,1,1/n);
        else
            x_mm = randn(n,1);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - - - MM approach (analytic proximal map) - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % should set rho to sufficiently high value,
    % though we can always just let the algorithm ratchet up the rho
    % that is the approach used here
    rho = 1;
    
    % initialize other problem parameters for MM algorithm
    epsilon       = 1;
    epsilon_reset = 0;
    current_obj   = Inf;
    
    % start timer
    tic;
    
    % want to map X, Y to A, b
    A = X' * X;
    b = - X' * Y;
    
    % algorithmic quirk is that we will optimize on X'X, but lasso operates
    % on X. apply svd to X, use to form spectral decomposition of A = X'X
    [~,S,V] = svd(full(X));
    
    if m > n
        s = diag(S); % these are singular values of X
    else
        s = [diag(S); zeros(n-m,1)]; % add extra zero eigenvalues of A
    end
    
    d = s.^2;    % eigenvalues of A
    
    % can now specify maximum rho as in quadratic program
    rho_max = (2 * d(end) / d(1) + 1) * norm(b,2);
    
    % project onto k largest components, then calculate loss, objective
    xk = project_k(x_mm, k);
    next_loss = 0.5*x_mm'*A*x_mm + b'*x_mm;
    next_obj = next_loss ...
        + rho * sqrt(sum_square(x_mm - xk) + epsilon);

    % uncomment for formatted output to monitor algorithm progress
%     fprintf('\nBegin MM algorithm\n\n'); 
%     fprintf('Iter\tNorm\tFeasible Dist\tRho\tEpsilon\tObjective\n');

    % main loop
    for mm_iter = 1:max_iter
 
        % notify and break if maximum iterations are reached.
        if(mm_iter >= max_iter)
            fprintf(2, 'MM algorithm has hit maximum iterations %d!\n', ...
                mm_iter);
            fprintf(2, 'Current Objective: %3.10f\n', current_obj);
            
            % stop timer
            mm_stop = toc;
            
            % send elements below tolerance to zero
            x_mm(abs(x_mm) < tolerance) = 0;
            
            % calculate "final" loss, objective
            mm_loss = full(0.5*x_mm'*A*x_mm + b'*x_mm);
            mm_obj =  mm_loss ...
              + rho * sqrt(sum_square(x_mm - project_k(x_mm,k)) + epsilon);

            % these are output variables for function
            MM_time = mm_stop;
            MM_opt = mm_obj;
            MM_iter = mm_iter;
            return;
        end
        
        % save previous iterate and previous objective function value.
        x_0 = x_mm;
        current_obj = next_obj;

        % update step size, iterate, objective function value
        lam = sqrt( sum_square(x_0 - project_k(x_0,k)) + epsilon) / rho;

        % naive proximal map is
        % inv(A + I/lam) (v/lam - b)
%         x_mm = prox_quad(xk, lam, A, b);

        % but we want to exploit structure in A! use spectral decomposition
        % of A (svd of X) to get A = VDV' and then calculate
        % U inv(D^2 + I/lam) V' (v/lam - b)
        % using O(n^2) mat-vec operations
        
        %%%% 1 ./ (diag(D' * D) + 1/lam)) .* z;
        z = xk - b * lam;
        z = V' * z;
        z = (1 ./ (lam * d + 1)) .* z;
        x_mm = V * z;

        % calculations for constraints, convergence
        xk = project_k(x_mm, k);
        dist = norm(x_mm - xk, 2);

        next_loss = 0.5*x_mm'*A*x_mm + b'*x_mm;
        next_obj =  next_loss ...
            + rho * sqrt(dist^2 + epsilon);

        the_norm = norm(x_mm - x_0, 2);
        scaled_norm = the_norm / ( norm(x_0, 2) + 1);
        converged = scaled_norm < tolerance;
        feasible = dist < tolerance;

        % output iteration information.
%         fprintf('%d\t%3.7f\t%3.7f\t%3.4f\t%1.12f\t%3.7f\n', ...
%             mm_iter, the_norm, dist, rho, epsilon, next_obj);


        % check for convergence. if converged, then quit.
        if(feasible && converged)
            
            % at this point, algorithm converged before maximum iteration
            % output results to console
%             fprintf('\nMM algorithm has converged successfully.\n');
            mm_stop = toc;
            
            % send elements below tolerance to zero
            x_mm(abs(x_mm) < tolerance) = 0;
            mm_loss = full(0.5*x_mm'*A*x_mm + b'*x_mm);
            mm_obj =  mm_loss ...
                + rho * sqrt(sum_square(x_mm - project_k(x_mm,k)) + epsilon);
%             fprintf('MM Results:\nIterations: %d\n', mm_iter);
%             fprintf('Final rho: %3.4f\n', rho);
%             fprintf('Final Loss: %3.10f\n', mm_loss);
%             fprintf('Final Objective: %3.10f\n', mm_obj);
%             fprintf('Sparsity constraint satisfied to tolerance %3.10f? %d\n', ...
%                 tolerance, full(feasible));
%             fprintf('Total Compute Time: %3.7f\n', mm_stop );

            MM_time = mm_stop;
            MM_opt = mm_obj;
            MM_iter = mm_iter;
            return;
        end

        % if in feasible set, and epsilon has not been reset to 1 before,
        % then reset epsilon and stop incrementing rho
        if(feasible && epsilon_reset < 1)
            epsilon = eps_dec;
            epsilon_reset = epsilon_reset + 1;
            rho_inc = 1;
        end
        
        % algorithm is unconverged at this point.
        % if algorithm is in feasible set, then rho should not be changing
        % check descent property in that case
        % if rho is not changing but objective increases, then abort
        if(feasible && next_obj > current_obj + tolerance)
            fprintf(2, '\nMM algorithm fails to descend!\n');
            fprintf(2, 'MM Iteration: %d \n', mm_iter);
            fprintf(2, 'Current Objective: %3.10f \n', current_obj);
            fprintf(2, 'Next Objective: %3.10f \n',next_obj);
            fprintf(2, 'Difference in objectives: %3.10f\n', ...
                abs(next_obj - current_obj));
            return;
        end
        
        % check for infeasibility. if infeasible, increment rho and
        % decrement epsilon
        if(dist < 1e-4 || the_norm < 1e-4)
            rho = min(rho * rho_inc, rho_max);
            epsilon = max(epsilon_decrement(epsilon, ...
              sum_square(x_mm - project_k(x_mm,k)), 10, eps_dec), ...
              epsilon_min);
        end
        
    end % end main loop
end % end function
    
% function x = prox_quad(v, lambda, A, b)
% % PROX_QUAD    The proximal operator of a quadratic function
% %
% %   f(x) = 0.5*x'Ax + b'x
% %   prox_quad(v,lambda,A,b)
% % Coded by Neal Parikh (2013). If used, then please cite the reference
% %    Parikh N, Boyd S (2013) Proximal algorithms. 
% %    _Foundations Trends Optimization_ *1*:123?231.
% % 
% 
%     rho = 1/lambda;
%     m = size(A);
%     if issparse(A)
%         x = (A + rho*speye(m)) \ (rho*v - b);
%     else
%         x = (A + rho*eye(m)) \ (rho*v - b);
%     end
% end

% function y = epsilon_decrement(eps, norm, first_decrement, second_decrement)
% % EPSILON DECREMENT Two-regime decrementer for epsilon
% % 
% % This function decrements epsilon with two scales.
% % The first scale occurs when NORM is of greater order than EPS.
% % The second scale occurs when EPS is of equivalent or greater scale than
% % NORM.
% %
% % epsilon_decrement(eps, norm, first_decrement, second_decrement)
% 
%     if(log(eps) < log(norm))
%         y = eps / first_decrement;
%     else
%         y = eps / second_decrement;
%     end
% end

% function y = project_k(x, k)
% % PROJECT ONTO K LARGEST NONZERO ELEMENTS
% %
% % This function projects a vector X onto its K elements of largest
% % magnitude. project_k can handle both integer and double input, but it
% % does not check for repeated values. Consequently, if a tie occurs,
% % project_k will rely on the indices given by descending sort and then
% % truncate at K values.
%     n = length(x);
%     y = zeros(size(x));
%     if(k > n)
%         error('Argument k exceeds dimension of vector!');
%     elseif (k <= 0)
%         error('Argument k to project_k() is not positive.');
%     else
%         [~, indies] = sort(abs(x), 'descend');
%         y(indies(1:k)) = x(indies(1:k));
%     end
% end
