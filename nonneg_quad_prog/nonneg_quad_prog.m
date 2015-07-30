function [MM_iter, MM_time, MM_opt, CVX_iter, CVX_time, CVX_opt, ...
    m_stop, m_obj, Y_time, Y_opt] = nonneg_quad_prog(A, b)
% NONNEGATIVE QUADRATIC PROGRAMMING
% This function solves the optimization problem
%   minimize 0.5*x'Ax + b'x
%   subject to x >= 0,
% by exact penalization and proximal mapping. It uses two MM algorithms, 
% one with an analytic proximal map and the other with a Landweber
% iteration, to solve the problem. It also solves the problem with CVX and YALMIP for
% a basis of comparison.
%
% Arguments:
%
% -- A is the design matrix. It *MUST* be positive semidefinite or else CVX
%    will kill the process!
% -- b is the response vector. There are no restrictions on b.
%
% Coded by Kevin L. Keys (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    % global parameters
    tolerance   = 1e-6;
    max_iter    = 1e4;
    epsilon_min = 1e-15;
    rho_inc     = 1.1;
    eps_dec     = 1.2;
    n           = length(b);


    % initialize all output variables to zero.
    CVX_iter    = 0;
    MM_iter     = 0;
    CVX_time    = 0;
    MM_time     = 0;
    CVX_opt     = 0;
    MM_opt      = 0;
    Y_time      = 0;
    Y_opt       = 0;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - - - MM approach (analytic proximal map) - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % should set rho to sufficiently high value,
    % though we can always just let the algorithm ratchet up the rho
    % that is the approach used here
    rho = 1;
    
    % initialize other problem parameters for MM algorithm
    epsilon       = 1;
    mm_iter       = 0;
    epsilon_reset = 0;
    if issparse(A)
        x_mm = sprandn(n,1,1/n);
    else
        x_mm = randn(size(b));
    end
    current_obj = Inf;

    % start timer
    tic;
    
    % apply spectral decomposition to A
    [V,D] = eig(A);
    d     = diag(D);
    clear D; % D is no longer needed, so clear it from memory
    
    % approximate Lipschitz constant for capping rho
    rho_max = (2 * d(end) / d(1) + 1) * norm(b,2);
%     fprintf('Condition number on A: %3.6f\n', d(end) / d(1));
    
    next_loss = 0.5*x_mm'*A*x_mm + b'*x_mm;
    next_obj  = next_loss ...
                + rho * sqrt(sum_square(x_mm - max(x_mm,0)) + epsilon);
    in_box    = all(x_mm > - tolerance);

    % uncomment for formatted output
%     fprintf('\nBegin MM algorithm\n\n'); 
%     fprintf('Iter\tNorm\tMin(x)\tRho\tEpsilon\tObjective\n');

    % main MM loop
    for mm_iter = 1:max_iter
 
            % notify and break if maximum iterations are reached
        if(mm_iter >= max_iter)
            fprintf(2, 'MM algorithm has hit maximum iterations %d!\n', ...
                mm_iter);
            fprintf(2, 'Current Objective: %3.10f\n', current_obj);

            % stop timer and do "final" calculations
            mm_stop = toc;
            mm_loss = full(0.5*x_mm'*A*x_mm + b'*x_mm);
            mm_obj =  mm_loss ...
                + rho * sqrt(sum_square(x_mm - max(x_mm,0)) + epsilon);
            fprintf(2, 'MM Results:\nIterations: %d\n', mm_iter);
            fprintf(2, 'Final rho: %3.4f\n', rho);
            fprintf(2, 'Final Loss: %3.10f\n', mm_loss);
            fprintf(2, 'Final Objective: %3.10f\n', mm_obj);
            fprintf(2, 'Box constraint satisfied to tolerance %3.10f? %d\n', ...
                tolerance, full(in_box));
            fprintf(2, 'Total Compute Time: %3.7f\n', mm_stop );

            MM_time = mm_stop;
            MM_opt = mm_obj;
            MM_iter = mm_iter;

            break;
        end
        
        % save previous iterate and previous objective function value.
        x_0         = x_mm;
        current_obj = next_obj;
        
        % this problem can only attain good accuracy with excruciatingly
        % gently increments on rho. after some number of iterations (say,
        % 20), the algorithm must stop trying to enforce constraints and
        % instead decrease the loss function
        if(mm_iter == 20)
            rho_inc = 1.001;
        end
        
        % update step size
        lam = sqrt( sum_square(x_0 - max(x_0,0)) + epsilon) / rho;

        % naive proximal map is
        % inv(A + I/lam) (v/lam - b)
        % can obtain result with one line:
        %
        % x_mm = prox_quad(max(x_0,0), lam, A, b);
        %
        % but we want to exploit structure in A! apply a spectral
        % decomposition on A to get A = VDV' and then calculate
        % V inv(D + I/lam) V' (v/lam - b)
        % using O(n^2) mat-vec operations. code here is somewhat pedantic
        % and awkward, but it ensures mat-vec operations at each line
        
        z    = max(x_0,0) - b * lam;
        z    = V' * z;
        z    = (1 ./ (lam * d + 1)) .* z;
        x_mm = V * z;
        
        % update loss, obj
        next_loss = 0.5*x_mm'*A*x_mm + b'*x_mm;
        next_obj  =  next_loss ...
                    + rho * sqrt(sum_square(x_mm - max(x_mm,0)) + epsilon);

        % calculations for constraints, convergence
        in_box      = all(x_mm >= - tolerance);
        the_norm    = norm(x_mm - x_0, 2);
        scaled_norm = the_norm / ( norm(x_0, 2) + 1);
        converged   = scaled_norm < tolerance;

        % output iteration information.
%         fprintf('%d\t%3.7f\t%3.7f\t%3.4f\t%1.12f\t%3.7f\n', ...
%             mm_iter, the_norm, full(min(x_mm)), rho, epsilon, next_obj);


        % check for convergence. if converged, then exit MM loop
        if(in_box && converged)  
            fprintf('\nMM algorithm has converged successfully.\n');
            
            % stop timer
            mm_stop = toc;
            
            % calculate final loss, obj
            mm_loss = full(0.5*x_mm'*A*x_mm + b'*x_mm);
            mm_obj =  mm_loss ...
                    + rho * sqrt(sum_square(x_mm - max(x_mm,0)) + epsilon);
                
            fprintf('MM Results:\nIterations: %d\n', mm_iter);
            fprintf('Final rho: %3.4f\n', rho);
            fprintf('Final Loss: %3.10f\n', mm_loss);
            fprintf('Final Objective: %3.10f\n', mm_obj);
            fprintf('Box constraint satisfied to tolerance %3.10f? %d\n', ...
                tolerance, full(in_box));
            fprintf('Total Compute Time: %3.7f\n', mm_stop );

            MM_time = mm_stop;
            MM_opt = mm_obj;
            MM_iter = mm_iter;
            break;
        end

        % algorithm is not yet converged
        % if in feasible set, and epsilon has not been reset to 1 before,
        % then reset epsilon and stop incrementing rho
        if(in_box && epsilon_reset < 1)
            epsilon = eps_dec;
            epsilon_reset = epsilon_reset + 1;
            rho_inc = 1;
        end
        
        % algorithm is unconverged at this point.
        % if algorithm is in feasible set, then rho should not be changing
        % check descent property in that case
        % if rho is not changing but objective increases, then abort
        if(in_box && next_obj > current_obj + tolerance)
            fprintf(2, '\nMM algorithm fails to descend!\n');
            fprintf(2, 'MM Iteration: %d \n', mm_iter);
            fprintf(2, 'Current Objective: %3.10f \n', current_obj);
            fprintf(2, 'Next Objective: %3.10f \n',next_obj);
            fprintf(2, 'Difference in objectives: %3.10f\n', abs(next_obj - current_obj));
            return;
        end

        % algorithm is still unconverged at this point
        % algorithm must now iterate, so update epsilon and rho
        % ratchet down epsilon
        epsilon = max(epsilon_decrement(epsilon, ...
                  sum_square(x_mm - max(x_mm,0)), eps_dec, eps_dec), ...
                  epsilon_min);

        % check for infeasibility. if infeasible, increment rho
        % ignore if algorithm is "close" to feasibility
        if(any(x_mm < -1e-4) || the_norm > 1e-4)
            rho = min(rho * rho_inc, rho_max);
        end
        
    end
    
    % reduce memory overhead by clearing variables that only MM uses
    clear V d z x_mm x_0
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - - - MATLAB approach - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % this approach uses MATLAB's native quadratic programming routine
    fprintf('\nCalculate with MATLAB''s Optimization Toolbox\n');
    
    % start timing
    tic;
    
    % solve quadratic program
    % order of arguments is
    % -- quadratic objective parameters, i.e. A,b
    % -- matrix and vector for inequality constraints; empty in this case
    % -- matrix, vector for equality constraints; also empty here
    % -- bound constraints for design variable z; only nonnegativity for us
    [z, fval, exitflag] = quadprog(A,b, [], [], [], [], zeros(n,1), []);
    
    if(exitflag == 1)
        z_stat = 'Function converged to a solution x.\n';
    elseif (exitflag == 0)
        z_stat = 'Number of iterations exceeded MaxIter.\n';
    elseif (exitflag == -2)
        z_stat = 'No feasible point was found.\n';
    elseif (exitflag == -3)
        z_stat = 'Problem is unbounded.\n';    
    elseif (exitflag == -4)
        z_stat = 'NaN value was encountered during execution of the algorithm.\n';
    elseif (exitflag == -5)
        z_stat = 'Both primal and dual problems are infeasible.\n';
    else
        z_stat = 'Search direction became too small. No further progress could be made.\n';
    end
    
    % stop timer
    m_stop = toc;
    
    % check constraint satisfaction, objectives
    in_box_m = all(z > - tolerance);
    m_loss   = fval;
    m_obj    = m_loss + rho * sqrt(sum_square(z - max(z,0)) + epsilon);
        
    fprintf('\nMATLAB quadprog() Results:\n');
    fprintf('Final Loss: %3.10f\n', m_loss);
    fprintf('Final Objective: %3.10f\n', m_obj);
    fprintf(z_stat);
    fprintf('Box constraint satisfied to tolerance %3.10f? %d\n', ...
        tolerance, full(in_box_m));
    fprintf('Total Compute Time: %3.7f\n', m_stop);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  - - - CVX approach - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % this operates on the LOSS FUNCTION, not on the OBJECTIVE FUNCTION
    % CVX cannot handle sqrt(dist(x,S) + eps) due to DCP rules.
    % use this method to get the minimum, and then add the penalty term
    % later (in printed output). Note that CVX *CAN* handle the convex
    % constraint, so in some respect the CVX answer should be better than
    % the MM one.

    cvx_precision low;
    cvx_begin quiet
        variable x(n) nonnegative;
        minimize (0.5*x'*A*x + b'*x);
    cvx_end

    % print results to console
    in_box_cvx = all(x > - tolerance);
    fprintf('\nCVX Results:\n');
    fprintf('Iterations: %d\n', cvx_slvitr);
    fprintf('Final loss: %3.10f\n', cvx_optval);
    fprintf('Final Objective: %3.10f \n', ... 
        cvx_optval + rho * sqrt(sum_square(x - max(x,0)) + epsilon));
    fprintf('Time: %3.3f\n', cvx_cputime);
    fprintf('Satisfies box constraint? %d\n', in_box_cvx);

    % save these values to return as function
    CVX_time = cvx_cputime;
    CVX_iter = cvx_slvitr;
    CVX_opt = cvx_optval;

    % clear CVX variables
    clear x;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % - - - YALMIP approach - - - %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    yalmip('clear');
    tic;
    y = sdpvar(n,1);
    Constraints = [y >= 0];
    Objective   = 0.5*y'*A*y + b'*y;
    options     = sdpsettings('verbose',0,'solver','mosek');
    sol = solvesdp(Constraints,Objective,options);
    if sol.problem == 0
        solution = double(y);
    else
        display('Hmm, something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end

    Y_time = toc;
    y_loss = 0.5*solution'*A*solution + b'*solution;
    Y_opt  = y_loss ...
            + rho * sqrt(sum_square(solution - max(solution,0)) + epsilon);
    in_box_y = all(solution > - tolerance);
    fprintf('\nYALMIP Results:\n');
    fprintf('Final Loss: %1.7f\n', y_loss);
    fprintf('Final Objective: %1.7f \n', Y_opt);
    fprintf('Time: %1.7f\n', Y_time);
    fprintf('Satisfies constraints? %d\n', in_box_y);
    
end
%     
% % function x = prox_quad(v, lambda, A, b)
% % % PROX_QUAD    The proximal operator of a quadratic function
% % %
% % %   f(x) = 0.5*x'Ax + b'x
% % %   prox_quad(v,lambda,A,b) 
% % 
% %     rho = 1/lambda;
% %     m = size(A);
% %     if issparse(A)
% %         x = (A + rho*speye(m)) \ (rho*v - b);
% %     else
% %         x = (A + rho*eye(m)) \ (rho*v - b);
% %     end
% % end
% 
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