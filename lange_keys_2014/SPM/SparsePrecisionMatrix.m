function [X,d] = SparsePrecisionMatrix(S,k,S0)
% SPARSE PRECISION MATRIX INFERENCE
% This function fits a positive semidefinite precision matrix X with at 
% most k nonzero upper triangular entries to a sample covariance matrix S. 
% It solves the optimization problem
%
%    min  -logdet(X) + tr(SX)
%    s.t. X in Omega,
%
% where Omega is the set of symmetric positive definite matrices with at
% most k nonzero entries in its upper triangle. The k entries are found as
% the largest k such entries by magnitude.
%
% Arguments:
% -- S is the sample covariance matrix.
% -- k is the number of desired elements in the upper triangle of X.
% -- S0 is a precision matrix supplied as a warm-start.
%
% Coded by Kevin L. Keys and Kenneth Lange (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    % global parameters
    tolerance = 1e-6;
    max_iter  = 1e3;
    eps_min   = 1e-15;
    eps_dec   = 1.2;
    rho_inc   = 1.2;
    d         = eig(S);
    d         = 1 ./ abs(d); % set components of d to their reciprocals
    rho_max   = 10 * (norm(d, 2) + norm(S, 'fro'));
    
    % can warm start with precision matrix S0
    if exist('S0','var') && ~isempty(S0)
        X       = S0;
        d       = eig(X);
        P       = ProjectSparseSymmetric(X,k);
        dist    = norm(X - P, 'fro');
        epsilon = 1;
        rho     = 1;
        
        % start loss, objective at lower values
        mm_loss     = -sum(log(d)) + trace(S*X);
        current_obj = - log(sum(d)) + trace(S*X) ...
            + rho * sqrt(dist^2 + epsilon);
    else
%         fprintf('no warm start\n');
        X       = randn(size(S));
        dist    = 1e6;
        rho     = 1;
        epsilon = 1;
        
        % naive starting values for loss, obj
        current_obj = Inf;
        mm_loss     = Inf;
    end
    
    % start timer
    tic;
    
%     % formatted output to monitor algorithm progress
%     fprintf('\nBegin MM algorithm\n\n'); 
%     fprintf('Iter\tDistance\tRho\tEpsilon\t\tLoss\tObjective\n');
%     fprintf('%d\t%3.7f\t%3.7f\t%3.9f\t%3.7f\t%3.7f\n', ...
%         0, dist, rho, epsilon, mm_loss, current_obj);
    
    % main loop
    for mm_iter = 1:max_iter
        
        % notify and break if maximum iterations are reached.
        if(mm_iter >= max_iter)
            fprintf(2, 'MM algorithm has hit maximum iterations %d!\n', mm_iter);
            fprintf(2, 'Current Objective: %3.10f\n', current_obj);
            
            % stop timer
            mm_stop = toc;
            
            % send elements below tolerance to zero
            X(abs(X) < tolerance) = 0;
            
            % calculate "final" loss, objective
            mm_loss = -sum(log(d)) + trace(S*X);
            mm_obj  =  mm_loss ...
                    + rho * sqrt(norm(X - P, 'fro') + epsilon);

%             fprintf(2, 'MM Results:\nIterations: %d\n', mm_iter);
%             fprintf(2, 'Final rho: %3.4f\n', rho);
%             fprintf(2, 'Final Loss: %3.10f\n', mm_loss);
%             fprintf(2, 'Final Objective: %3.10f\n', mm_obj);
%             fprintf(2, 'Sparsity constraint satisfied to tolerance %3.10f? %d\n', ...
%                 tolerance, dist < tolerance);
%             fprintf(2, 'Total Compute Time: %3.7f\n\n', mm_stop);
            return;
        end
        
        % project to a sparse symmetric matrix and then calculate distance
        % to feasibility, weight
        P    = ProjectSparseSymmetric(X,k);
        dist = norm(X - P, 'fro');
        w    = rho / sqrt(dist^2 + epsilon);
        
        % extract spectral decomposition of perturbed sample covar. matrix.
        [U,D] = eig(S - w*P);
        d     = diag(D);
        
        % overwrite eigenvalues with new ones as calculated by quadratic
        % formula
        d = (-d + sqrt(d.^2 + 4*w)) / (2*w);

        % construct new matrix X with updated eigenvalues d, and then
        % update the loss and objective values with new X
        X        = U * diag(d) * U';
        mm_loss  = -sum(log(d)) + trace(S*X);
        next_obj = mm_loss + rho * sqrt(dist^2 + epsilon);
        
%         % formatted output to monitor algorithm progress
%         fprintf('%d\t%3.7f\t%3.7f\t%3.9f\t%3.7f\t%3.7f\n', ...
%         mm_iter, dist, rho, epsilon, mm_loss, next_obj);

        % is algorithm in feasible set? is it converged?
        feasible  = dist < tolerance;
        converged =  feasible && abs(current_obj - next_obj) < tolerance;

        % check for convergence. if converged, then quit
        if converged
            
%             fprintf('\nMM algorithm has converged successfully.\n');
            % all elements smaller than tolerance sent to zero
            X(abs(X) < tolerance) = 0;
            
            % stop timer
            mm_stop = toc;
            
            % update loss
            mm_loss = -sum(log(d)) + trace(S*X);
            
            % perform final projection and then update loss
            P      = ProjectSparseSymmetric(X,k);
            mm_obj = mm_loss ...
                     + rho * sqrt(norm(X - P, 'fro') + epsilon);

%             fprintf('MM Results:\nIterations: %d\n', mm_iter);
%             fprintf('Final rho: %3.4f\n', rho);
%             fprintf('Final Loss: %3.10f\n', mm_loss);
%             fprintf('Final Objective: %3.10f\n', mm_obj);
%             fprintf('Sparsity constraint satisfied to tolerance %3.10f? %d\n', ...
%                 tolerance, dist < tolerance);
%             fprintf('Total Compute Time: %3.7f\n\n', mm_stop);
            return;
        end
        
        % algorithm unconverged at this point
        % if in feasible set, enforce descent property
        if(feasible && next_obj > current_obj + tolerance)
            fprintf(2, '\nMM algorithm fails to descend!\n');
            fprintf(2, 'MM Iteration: %d \n', mm_iter);
            fprintf(2, 'Current Objective: %3.10f \n', current_obj);
            fprintf(2, 'Next Objective: %3.10f \n',next_obj);
            fprintf(2, 'Difference in objectives: %3.10f\n', abs(next_obj - current_obj));
            return;
        end
        
        % at this point, algorithm is unconverged and infeasible
        % if algorithm is not _close_ to feasibility, then increment rho
        % ignore increment if algorithm is close to feasibility
        if dist > 100 * tolerance || ...
            abs(current_obj - next_obj) > 100 * tolerance
            rho = min(rho_inc * rho, rho_max);
        end
        
        % ratchet down epsilon at each iteration
        epsilon = max(epsilon / eps_dec, eps_min);

        % save objective function for next iteration
        current_obj = next_obj;
    end
end

function  Y = ProjectSparseSymmetric(X,k)
     % Projects a square matrix X onto the set of symmetric 
     % matrices with at most k upper triangular nonzero entries.
     
     % get matrix dimensions
     [m,n] = size(X);
     
     % dimensions should be the same
     if m ~= n
         error('Matrix X requires m == n!');
     end
     
     % k must not exceed # of elements in upper triangle
     if k > 0.5 * m * (m - 1)
         error('k exceeds number of entries in upper triangle of X.');
     end
     
     % store the diagonal
     d = diag(X);
     
     % zero out lower triangle
     Z = triu(X,1);
     
     % convert to a vector
     x = reshape(Z,m*n,1);
     
     % sort entries by magnitude
     [~,idx] = sort(abs(x));
     
     % preserve the sorted entries
     y = x(idx); 
 
     % zero all but k entries
     y(1:length(y)-k) = 0;
     
     % restore order
     y(idx) = y;
     
     % convert to a matrix
     Z = reshape(y,m,n);
     
     % convert to symmetric matrix
     Y = Z + Z' + diag(d); 
end
