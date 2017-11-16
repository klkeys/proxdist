function [MM_iter, MM_time, MM_opt, CVX_iter, CVX_time, CVX_opt, ...
    Y_time, Y_opt, diffs, fully_converged] = piecewiselin_over_binary(A,b)
% EXACT PENALIZATION OF PIECEWISE LINEAR LOSS FUNCTION OVER BINARY CONSTRAINTS
% 
% This function optimizes a piecewise linear loss function over the
% binary set $x \in \{0,1\}^n$. The precise problem statement is
%    min. sum_{j \neq i} A_{ij} |x_i - x_j| 
%    s.t.  x_i = 0 or x_i = 1, i = 1, 2, ... n
%
% For a reference, see notes for Lecture 9, UCLA EE236C, 
% by Lieven Vandenberghe: http://www.seas.ucla.edu/~vandenbe/ee236c.html. 
% If the algorithm does not return some x \in \{ 0, 1 \}^n,
% then we can round the relaxed solution to the nearest integer (here, 1/2
% will round down to zero) to obtain a binary optimal solution.
%
% Coded by Kevin L. Keys (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    % global parameters
    tolerance   = 1e-5;
    max_iter    = 2e2;
    epsilon_min = 1e-15;
    
    % vectors for box constraints
    n = length(b);
    l = zeros(n,1);
    u = ones(n,1);
    
    % initialize variables to store iterations, cpu time, optimal values
    CVX_iter        = 0;
    MM_iter         = 0;
    CVX_time        = 0;
    MM_time         = 0;
    Y_time          = 0;
    CVX_opt         = 0;
    MM_opt          = 0;
    Y_opt           = 0;
    diffs           = 0;
    fully_converged = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - - - MM approach - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%

    % start timer
    tic;

    % reset problem parameters for MM algorithm
    rho_max       = sum(sqrt(sum(A.^2))) + norm(b,2); % Lipschitz constant
    rho           = 1;
    rho_inc       = 1.2;
    eps_dec       = 1.2;
    epsilon       = eps_dec;
    epsilon_reset = 0;

    % initialize iterates and calculate first objective values
    x_mm        = 0.5 * ones(n,1) + randn(n,1);
    x_binary    = project_binary(x_mm);
    x_0         = Inf*ones(size(x_mm));
    current_obj = Inf;
    next_obj    = pairwise_differences(x_mm, A, b) ...
        + rho * sqrt(sum_square(x_mm - x_binary) + epsilon);

    % formatted output to monitor algorithm progress
%     fprintf('\nBegin MM algorithm\n\n'); 
%     fprintf('Maximum rho set to %3.6f\n', rho_max);
%     fprintf('Iter\tProx Iter\tScaled Norm\tRho\tEpsilon\tObjective\n');

    % main MM loop
    for mm_iter = 1:max_iter

        % notify and break if maximum iterations are reached.
        if(mm_iter >= max_iter)
            fprintf(2, 'MM algorithm has hit maximum iterations %d!\n', ...
                mm_iter);
            
            % stop timer
            mm_stop  = toc;
            
            % apply final binary projection
            x_binary = project_binary(x_mm);
            x_mm     = x_binary; % rounding solution!
            
            % calculate "final" loss, obj
            mm_loss  = pairwise_differences(x_mm,A,b);
            mm_obj   =  mm_loss ...
                    + rho * sqrt(sum_square(x_mm - x_binary) + epsilon);
                
            % print output
            fprintf(2, 'MM Results:\nIterations: %d\n', mm_iter);
            fprintf(2, 'Final rho: %3.4f\n', rho);
            fprintf(2, 'Final Loss: %3.7f\n', mm_loss);
            fprintf(2, 'Final Objective: %3.7f\n', mm_obj);
            fprintf(2, 'Binary constraint satisfied to tolerance %3.10f? %d\n', ...
                tolerance, in_binary);
            fprintf(2, 'Total Compute Time: %1.3f\n', mm_stop);
            fprintf(2, 'Total nonzeros in iterate: %d\n\n', sum(x_binary));

            MM_time  = mm_stop;
            MM_opt   = mm_obj;
            MM_iter  = mm_iter;
            break;
        end
        
        % guard against infinite looping caused by inability of MM to drive
        % iterate all the way to zero vector. If this happens, then kill
        % the MM algorithm and return progress up to that point. NOTA BENE:
        % the choice of the iterate at which to kill the algorithm is chosen 
        % somewhat arbitrarily, but 20 seems to work well in practice
        if ((sum(project_binary(x_0)) == 0 && sum(project_binary(x_mm)) == 0) ...
           || (sum(project_binary(x_0)) == n && sum(project_binary(x_mm)) == n)) ...
           && mm_iter > 20
            fprintf(2, 'WARNING: MM iterate projects to either zero or one vector!\n');
            fprintf(2, 'Usually indicates very small weights in parameters A or b.\n');
            fprintf(2, 'Algorithm unlikely to change iterate at this point, aborting...\n');
            
            % stop timer
            mm_stop  = toc;
            
            % degenerate solutions are 0 or 1
            % default to 1, and check if 0 returns lower loss
            x_mm     = ones(size(x_mm));
            x_binary = x_mm;
            x_alt    = zeros(size(x_mm));
            mm_loss  = pairwise_differences(x_mm,A,b);
            mm_alt   = pairwise_differences(x_alt,A,b);
            
            % if 0 returns lower loss, use that one as solution
            if(mm_alt < mm_loss)
                x_mm     = x_alt;
                mm_loss  = mm_alt;
                x_binary = x_alt;
            end
            
            % calculate "final" loss
            mm_obj  = mm_loss ...
                    + rho * sqrt(sum_square(x_mm - x_binary) + epsilon);
            MM_time = mm_stop;
            MM_opt  = mm_obj;
            MM_iter = mm_iter;
            
            fprintf('MM Results:\nIterations: %d\n', mm_iter);
            fprintf('Final rho: %3.4f\n', rho);
            fprintf('Final Loss: %3.7f\n', mm_loss);
            fprintf('Final Objective: %3.7f\n', mm_obj);
            fprintf('Binary constraint satisfied to tolerance %3.10f? %d\n', ...
                tolerance, in_binary);
            fprintf('Total Compute Time: %1.3f\n', mm_stop);
            fprintf('Total nonzeros in iterate: %d\n\n', sum(x_mm));
            break;
        end
        
        % save previous iterate and previous objective function value
        x_0         = x_mm;
        current_obj = next_obj;
        x_binary    = project_binary(x_0);
        
        % update step size lambda, iterate, objective function value
        lam = sqrt(sum_square(x_0 - x_binary) + epsilon) / rho;

        [x_mm, prox_iter] = prox_separable_KLK(x_0, lam, A, b, l, u, x_binary, 1e-8);
        x_binary          = project_binary(x_mm);
        
        next_loss = pairwise_differences(x_mm, A, b);
        next_obj  =  next_loss ...
                    + rho * sqrt(sum_square(x_mm - x_binary) + epsilon);

        % calculations for constraints, convergence
        in_binary   = all(abs(x_mm - x_binary) < tolerance) && ...
                      all(abs(x_0 - project_binary(x_0)) < tolerance);
        the_norm    = norm(x_mm - x_0, 2);
        scaled_norm = the_norm / ( norm(x_0,2) + 1);
        converged   = scaled_norm < tolerance;
        fully_converged = converged && in_binary;
        
        % output iteration information.
%         fprintf('%d\t%d\t%3.7f\t%3.4f\t%1.12f\t%3.7f\n', ...
%             mm_iter, prox_iter, scaled_norm, rho, epsilon, next_obj);

        % check for convergence. if converged, then quit and output the
        % results from MM algorithm to console
        if(in_binary && converged)
            fprintf('\nMM algorithm has converged successfully.\n');
            
            % stop timer
            mm_stop  = toc;
            x_binary = project_binary(x_mm);
            x_mm     = x_binary;
            mm_loss  = pairwise_differences(x_mm,A,b);
            mm_obj   =  mm_loss ...
                    + rho * sqrt(sum_square(x_mm - x_binary) + epsilon);
            fprintf('MM Results:\nIterations: %d\n', mm_iter);
            fprintf('Final rho: %3.4f\n', rho);
            fprintf('Final Loss: %3.7f\n', mm_loss);
            fprintf('Final Objective: %3.7f\n', mm_obj);
            fprintf('Binary constraint satisfied to tolerance %3.10f? %d\n', ...
                tolerance, in_binary);
            fprintf('Total Compute Time: %1.3f\n', mm_stop);
            fprintf('Total nonzeros in iterate: %d\n\n', sum(x_binary));

            MM_time  = mm_stop;
            MM_opt   = mm_obj;
            MM_iter  = mm_iter;
            break;
        end

        % if in feasible set, and epsilon has not been reset to 1 before,
        % then reset epsilon and stop incrementing rho
        if(in_binary && epsilon_reset < 1)
            epsilon       = eps_dec;
            epsilon_reset = epsilon_reset + 1;
            rho_inc       = 1;
        end

        % algorithm is unconverged at this point.
        % if algorithm is in feasible set, then rho should not be changing
        % check descent property in that case
        % if rho is not changing but objective increases, then abort
        if(in_binary && next_obj > current_obj + tolerance)
            fprintf(2, '\nMM algorithm fails to descend!\n');
            fprintf(2, 'MM Iteration: %d \n', mm_iter);
            fprintf(2, 'Current Objective: %3.7f \n', current_obj);
            fprintf(2, 'Next Objective: %3.7f \n',next_obj);
            fprintf(2, 'Difference in objectives: %3.7f\n', ...
                abs(next_obj - current_obj));
            return;
        end

        % algorithm is still unconverged at this point
        % algorithm must now iterate, so ratchet down epsilon
        epsilon = max(epsilon_decrement(epsilon, sum_square(x_mm - x_binary), ...
                  eps_dec, eps_dec), epsilon_min);

        % check for infeasibility. if infeasible, increment rho
        % also increment if norm of iterates is relatively large
        if(scaled_norm > 1e-3 || any(abs(x_mm - x_binary)) > 1e-3)
            rho = min(rho * rho_inc, rho_max);
        end
        
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - - - CVX approach - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % this operates on the LOSS FUNCTION, not on the OBJECTIVE FUNCTION
    % CVX cannot handle sqrt(dist(x,S) + e) due to DCP rules
    % use this method to get the minimum, and then add the penalty term
    % later (in printed output).
    %
    % also note that CVX really suffers with large dimensions. to prevent
    % the machine from crashing, we will not let CVX calculate beyond n=512
    
    if(n <= 512)
        cvx_begin quiet
            cvx_solver gurobi;
            cvx_precision low;
            variable x(n) binary;
            minimize (pairwise_differences(x, A, b));
        cvx_end

        cvx_binary = project_binary(x);
        in_box_cvx = all(x - cvx_binary < tolerance);
        fprintf('CVX Results:\n');
        fprintf('Iterations: %d\n', cvx_slvitr);
        fprintf('Final Loss: %1.7f\n', cvx_optval);
        fprintf('Final Objective: %1.7f \n', ...
            cvx_optval + rho * sqrt(sum_square(x - cvx_binary) + epsilon));
        fprintf('Time: %1.3f\n', cvx_cputime);
        fprintf('Satisfies constraints? %d\n', in_box_cvx);
        fprintf('Total nonzeros in iterate: %d\n\n', sum(cvx_binary));

        CVX_time = cvx_cputime;
        CVX_iter = cvx_slvitr;
        CVX_opt  = cvx_optval;

        diffs = sum((cvx_binary - x_binary) ~= 0);
        fprintf('Number of different components between CVX, MM iterates: %d\n', ...
            diffs);
    end
        
 %%% 22 JAN 2014: so far YALMIP stutters in solving the binary problem
 %%% it can solve the box constraint just fine, but does so slowly
if 0
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - - - YALMIP approach - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    yalmip('clear');
    y = sdpvar(n,1);
%     Constraints = binary(y);
    Constraints = [y >= 0; y <= 1];
    Objective = 0.5 * trace(A * abs(y*ones(size(y))' - ones(size(y))*y')) + b' * y;
    options = sdpsettings('verbose',0,'solver','mosek');
    sol = solvesdp(Constraints,Objective,options);
    if sol.problem == 0
        
        % extract solution
        solution = double(y);
        
        % round solution
        solution = project_binary(solution);
        
        % now calculate output information
        y_loss   = pairwise_differences(solution,A,b);
        y_box    = project_box(solution, l, u);
        y_obj    = y_loss + rho * sqrt(sum_square(solution - y_box) + epsilon);
        in_box_y = all(abs(solution - y_box)) < tolerance;
        fprintf('YALMIP Results:\n');
        fprintf('Final Loss: %1.7f\n', y_loss);
        fprintf('Final Objective: %1.7f \n', y_obj);
        fprintf('Time: %1.7f\n', sol.yalmiptime);
        fprintf('Satisfies constraints? %d\n', in_box_y);

        Y_opt  = y_obj;
        Y_time = sol.yalmiptime;
    else
        display('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end
end

end

% function [x, iter] = prox_separable_KLK(v, lambda, A, b, l, u, x0, tol, MAX_ITER)
% % PROX_SEPARABLE_KLK   Evaluate the prox operator of a fully separable 
% %     function. This is Kevin's slightly recoded version of Neal Parikh's
% %     code from the PROXIMAL ALGORITHMS paper. IT is designed to handle
% %     piecewise linear loss functions of the form
% %     
% %     sum_{j > i} A_{ij} | x_i - x_j | + b^T x
% %
% % Arguments:
% %
% %  -- v is the point at which to evaluate the operator (projection).
% %  -- fp is a subgradient oracle for the function.
% %  -- lambda (optional) is the proximal parameter; defaults to 1.
% %  -- A is matrix of coefficients for the piecewise linear function.
% %  -- b is the linear coefficient of the piecewise linear function.
% %  -- k is the index for x on which we optimize the objective function.
% %  -- l (optional) is a lower bound for x; defaults to -Inf.
% %  -- u (optional) is an upper bound for x; defaults to Inf.
% %  -- x0 (optional) is a value at which to warm start the algorithm.
% %  -- tol (optional) is a stopping tolerance.
% %  -- MAX_ITER (optional) is the maximum number of iterations.
% %
% % coded by Neal Parikh (2013)
% % modified by Kevin L. Keys (2014)
% % klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% 
%     n = length(v);
% 
%     if ~exist('lambda', 'var') || isnan(lambda) || isempty(lambda)
%         lambda = 1;
%     end
%     rho = 1/lambda;
% 
%     if ~exist('l', 'var') || any(isnan(l)) || isempty(l)
%         l = -inf(n,1);
%     end
%     if ~exist('u', 'var') || any(isnan(u)) || isempty(u)
%         u = inf(n,1);
%     end
%     if ~exist('x0', 'var') || any(isnan(x0)) || isempty(x0)
%         x0 = zeros(n,1);
%     end
%     if ~exist('tol', 'var') || isnan(tol) || isempty(tol)
%         tol = 1e-10;
%     end
%     if ~exist('MAX_ITER', 'var') || isnan(MAX_ITER) || isempty(MAX_ITER)
%         MAX_ITER = 500;
%     end
% 
%     iter = 0;
%     x = max(l, min(x0, u));
% 
%     while any(u - l > tol) && iter < MAX_ITER
%         g = fp(x, v, A, b) + rho*(x - v);
% 
%         idx = (g > 0); 
%         l(idx) = max(l(idx), x(idx) - g(idx)/rho);
%         u(idx) = x(idx);
% 
%         idx = ~idx;
%         u(idx) = min(u(idx), x(idx) - g(idx)/rho);
%         l(idx) = x(idx);
% 
%         x = (l + u)/2;
%         iter = iter + 1;
%     end
% 
%     if any(u - l > tol)
%         fprintf(2, 'Warning: %d entries did not converge; max interval size = %f.\n', ...
%             sum(u - l > tol), max(u - l));
%         disp(x(u - l > tol));
%     end
%     
%     % subroutine to calculate subgradient oracle for prox_separable_KLK
%     function y = fp(x, xn, A, b)
%         
%         % first find pairwise averages of xn
%         pairwise_averages = 0.5 * bsxfun(@plus, xn, xn');
%    
%         % now subtract the pairwise averages of xn from the vector x,
%         my_diffs = bsxfun(@minus, x, pairwise_averages);
%         
%         % now perform Hadamard multiplication of A against sign(my_diffs), 
%         % and then sum along the rows and add b. 
%         % this gives the subgradient oracle.
%         % must have zeros either on diag(A) or diag(my_diffs)!
%         y = sum(A .* sign(my_diffs), 2) + b; 
%     end
% end
% 
% function y = pairwise_differences(x, A, b)
% % PIECEWISE LINEAR LOSS WITH BOX CONSTRAINTS FOR GRAPH CUT PROBLEM
% % 
% % This function calculates the piecewise linear function
% % \sum_{i > j} A_{ij} | x_i - x_j | + b^T x
% % subject to box constraints 0 <= x <= 1. These constraints constitute a
% % relaxation of the original binary constrained problem x \in \{ 0, 1 \}^n.
% % First find matrix of pairwise differences. then take
% % componentwise absolute values to get E. then perform A*E, which
% % serves to weight the absolute pairwise differences. the desired
% % sums lie on the diagonal of A*E. They are double-counted, 
% % so obtain the loss by calculating
% %
% % tr(AE)/2 + b^T x.
% %
% % CAUTION: The complexity of evaluating this function grows at least O(n^2)
% %
% % coded by Kevin L. Keys (2014)
% % klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%      y = 0.5 * trace(A * abs(x*ones(size(x))' - ones(size(x))*x')) + b' * x;
% 
% end