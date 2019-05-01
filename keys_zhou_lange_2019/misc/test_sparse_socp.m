function [] = test_sparse_socp()

    % set RNG
    seed = 2016;
    rng(seed);

    % set algorithm parameters
%     max_iter = 10000;
%     eps      = 1e-3;
%     quiet    = false;
%     verbose  = ~quiet;
%     inc_step = 100;
%     rho_inc  = 2.0;

    % set dimensions
    m = 1024;
    n = 2056;
    s = 0.01;
%     rho = 1/n;
    m = 4*m;
    n = 4*n;

    A = sprandn(m,n,s);
    x = sprand(n,1,s);
    c = sprand(n,1,s);
    b = sprand(m,1,s);
    d = norm(A*x + b);

    fprintf('Problem specs ok?\n');
    fprintf('Norm of A*x + b: %3.7f\n', norm(A*x + b));
    fprintf('dot(c,x) + d = %3.7f\n', dot(c,x) + d);
    fprintf('Feasible? %d\n', norm(A*x + b) <= dot(c,x) + d);

    w  = sprandn(n,1,s);
    fprintf('Before projection:\n');
    fprintf('norm(A*w + b): %3.7f\n', norm(A*w + b));
    fprintf('dot(c,w) + d: %3.7f\n', dot(c,w) + d);
    fprintf('Feasible? %d\n', norm(A*w + b) <= dot(c,w) + d);
    
    tic();
    pw = socp(w,A,b,c,d);
    mm_time = toc();
    
    fprintf('After projection:\n')
    fprintf('norm(A*pw + b): %3.7f\n', norm(A*pw + b));
    fprintf('dot(c,pw) + d: %3.7f\n', dot(c,pw) + d);
    fprintf('Feasible? %d\n', norm(A*pw + b) <= dot(c,pw) + d);

    tic;
    cvx_begin %quiet
        cvx_solver gurobi;
        cvx_precision low;
        variable u(n);
        minimize (0.5*sum_square(full(w) - u));
        subject to
            norm(A*u + b) <= dot(c,u) + d;  
    cvx_end
    cvx_time = toc();
    
    fprintf('Results:\n');
    fprintf('\t\tPD\tCVX\n');
    fprintf('Optima: %3.4f\t%3.4f\n', 0.5*norm(w-pw)^2, 0.5*norm(w-u)^2);
    fprintf('Time: %3.4f\t%3.4f\n', mm_time, cvx_time);
    fprintf('Distance b/w PD, CVX: %3.7f\n', norm(u-pw));

end
