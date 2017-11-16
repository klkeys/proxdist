function u = socp(x,A,b,c,d)
% PROJECTION ONTO SECOND ORDER CONE
%
% u = socp(x,A,b,c,d) computes a projection onto the second order cone defined by 
% 
% || Ax - b || <= c'*x + d.
%
% For affine constraint matrix A and sparse or dense vectors b and c, socp solves the optimization problem
%
%    minimize 0.5*norm(u - x) + lambda(Au - b)
%             u   >= 0
%
% with an accelerated proximal distance algorithm. The affine constraints 
% of the cone projection constitute part of the objective function.
% The vector `lambda` represents the Lagrange multiplier.

    % algorithm constants
    rho      = 1e-2;
    rho_inc  = 5.0;
    rho_max  = 1e30;
    max_iter = 10000;
    inc_step = 10;
    tol      = 1e-6;
    feastol  = 1e-6;
    quiet    = true;
    quiet    = false;
    q        = length(c);
    At       = A';
    u        = sprandn(q,1,0.1);
    u0       = u;
    loss    = 0.5*norm(x - u);
    dw      = Inf;
    dr      = Inf;

    % cache a factorization of I + rho*A'*A + rho*c*c'
    % we must update the factorization every time that we update rho
    I  = speye(q,q);
    AA       = At*A + c*c';
    AI = I + rho*AA;
    Ainv = chol(AI);
%    E        = [I; sqrt(rho)*A; sqrt(rho)*c'];
%    R = qr(E,0);

    if ~quiet
        fprintf('Iter\tLoss\tNorm\tdw\tdr\tRho\n');
        fprintf('%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n', 0, loss, Inf, dw, dr, rho);
    end
    
    for i = 1:max_iter

        % compute accelerated step z = y + (i - 1)/(i + 2)*(y - x)
        kx = (i - 1.0) / (i + 2.0);
        ky = 1.0 + kx;
        U = ky*u - kx*u0;
        u0 = u;

        % compute projections onto constraint sets;
        W  = A*U + b;
        r2 = dot(c,U) + d;
        [pw,pr] = proj_soc(W,r2);

        % update z for prox dist update
        % update b0 for Jacobi inversion
        z  = At*(b-pw) + (d-pr)*c;
        b0 = x - rho*z;

        % prox dist update
       u  = Ainv \ (Ainv' \ b0);
%       u  = R \ (R' \ b0);

        % now update w,r
        w = A*u + b;
        r = dot(c,u) + d;

        % convergence checks;
        loss        = 0.5*norm(u - x)^2;
        dw          = norm(w - pw);
        dr          = sqrt(abs(r*r - 2*r*pr + pr*pr));
        feas        = dr < feastol && dw < feastol;
        the_norm    = norm(u - u0);
        scaled_norm = the_norm / (norm(u0) + 1.0);
        converged   = scaled_norm < tol && feas;

        % print progress of algorithm;
        if (i <= 10 || mod(i,inc_step) == 0) && ~quiet
            fprintf('%d\t%3.7f\t%3.7f\t%3.7f\t%3.7f\t%3.7f\n', i, loss, the_norm, dw, dr, rho);
        end

        % if converged then break, else save loss and continue
        if converged
            u(abs(u) < tol) = 0.0;
            return;
        end
%         loss0 = loss;

        if mod(i,inc_step) == 0
            rho    = min(rho_inc*rho, rho_max);
            AI = I + rho*AA;
            Ainv = chol(AI);
%            E = [I; sqrt(rho)*A; sqrt(rho)*c'];
%            R = qr(E,0);
            u0 = u;
        end
    end

    % threshold small elements of y before returning;
    u(abs(u) < tol) = 0.0;
end

%%%%%%%%%%%%%%%%%%%
%%% subroutines %%%
%%%%%%%%%%%%%%%%%%%


function [z,u] = proj_soc(x,r)
%     if isempty(x)
%         z = [];
%         u = -Inf;
%         return;
%     elseif length(x) == 1
%         z = max(x,0);
%         u = 0.0;
%         return;
%     end

    n = norm(x);
    if n <= -r
        z = sparse(length(x),1);
        u = 0.0;
%         return;
    elseif n > abs(r)
        a = (n + r) / (2 * n);
        z = a*x;
        u = a*n;
%         return;
    else
        z = x;
        u = r;
    end
end

%test_dense_socp()
%test_sparse_socp()
