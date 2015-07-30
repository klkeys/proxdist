function [X,u,s,v] = MatrixCompletion(Y,W,r,varargin)
%
% This function approximates a matrix Y by a rank r matrix X.
% The missing pattern of Y is encoded in the 0/1 matrix W.
%
% Coded by Kevin L. Keys and Kenneth Lange (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    X = Y;
    
    % warm-start?
    if nargin > 3
        X = varargin{1};
    end
    if isempty(W) || ~exist('W', 'var')
        error('Missing pattern matrix W for missing data.');
    end
    if isempty(r) || ~exist('r', 'var')
        error('Missing rank constraint r.');
    end
    if isempty(Y) || ~exist('Y', 'var')
        error('Missing data matrix Y.');
    end
    
    % Global parameters
    rho_max   = full(6 * sum_square(Y(find(W))));
    tolerance = 1e-6;
    max_iter  = 1e3;
    eps_min   = 1e-15;
    [m,n]     = size(Y);
    
    if r >= min(m,n)
        fprintf(2, 'WARNING: rank constraint could exceed number of nonzero singular values.\n');
    end
    
    % Prepare for iterations.
    current_obj = 1e12;
    
    % reset problem parameters for MM algorithm
    rho     = 1;
    eps_dec = 1.2;
    epsilon = 1;
    rho_inc = 1.2;
    tic;
    
    % formatted output to monitor algorithm progress
%     fprintf('\nBegin MM algorithm\n\n'); 
%     fprintf('Iter\tDistance\tRho\tEpsilon\t     Objective\n');

    % main MM loop
    for mm_iter = 1:max_iter

        % notify and break if maximum iterations are reached.
        if(mm_iter >= max_iter)
            fprintf(2, 'MM algorithm has hit maximum iterations %d!\n', mm_iter);
            fprintf(2, 'Current Objective: %3.10f\n', current_obj);
            mm_stop = toc;
            mm_loss = 0.5 * norm(W .* (Y - X),'fro')^2;
            mm_obj =  mm_loss ...
                    + rho * sqrt(dist^2 + epsilon);
                
%             fprintf(2, 'MM Results:\nIterations: %d\n', mm_iter);
%             fprintf(2, 'Final rho: %3.4f\n', rho);
%             fprintf(2, 'Final Loss: %3.10f\n', mm_loss);
%             fprintf(2, 'Final Objective: %3.10f\n', full(mm_obj));
%             fprintf(2, 'Rank constraint satisfied to tolerance %3.10f? %d\n', ...
%                 tolerance, full(dist) < tolerance);
%             fprintf(2, 'Total Compute Time: %3.7f\n\n', mm_stop);
            break;
        end

        % save previous iterations, objective
        current_obj = next_obj;
        
        % update step size, iterate, objective function value
        Z       = W .* Y + (1 - W) .* X;
        [u,s,v] = ProjectLowRank(X,r);
        P       = u*s*v';
        dist    = norm(X - P, 'fro');
        w       = rho / sqrt(dist^2 + epsilon);
        X       = (Z + w*P) / (1 + w);
        
%         next_loss = 0.5 * norm(W .* (Y - X),'fro')^2;
        next_obj = 0.5 * norm(W .* (Y - X),'fro')^2 ...
                 + rho * sqrt(dist^2 + epsilon);
%         fprintf('%d\t%3.7f\t%3.4f\t%1.12f\t%3.7f\n', ...
%             mm_iter, full(dist), full(rho), full(epsilon), full(next_obj));

        % check for convergence. if converged, then quit and output the
        % results from MM algorithm to console
        if dist < tolerance && abs(current_obj - next_obj) < tolerance
%             fprintf('\nMM algorithm has converged successfully.\n');

            mm_stop = toc;
            mm_loss = 0.5 * norm(W .* (Y - X), 'fro')^2;
            mm_obj =  mm_loss ...
                    + rho * sqrt(dist^2 + epsilon);

%             fprintf('MM Results:\nIterations: %d\n', mm_iter);
%             fprintf('Final rho: %3.4f\n', rho);
%             fprintf('Final Loss: %3.10f\n', mm_loss);
%             fprintf('Final Objective: %3.10f\n', full(mm_obj));
%             fprintf('Sparsity constraint satisfied to tolerance %3.10f? %d\n', ...
%                 tolerance, full(dist) < tolerance);
%             fprintf('Total Compute Time: %3.7f\n\n', mm_stop);
            
            break;
        end
        
        % at this point, algorithm is not yet converged
        % increment rho if not yet close to feasibility
        if (distance > 100 * tolerance || ...
                abs(current_obj - next_obj) > 100 * tolerance)
            rho = min(rho_inc*rho, rho_max);
        end
        
        % increment epsilon at each iteration
        epsilon = max(epsilon/eps_dec, eps_min);
    end
end

% function  [U,S,V] = ProjectLowRank(X,k)
% % Projects a matrix X onto the closest matrix of rank k.
%     [U,S,V] = svds(X,k);
% %     Y = U*S*V';
% end