% run_NQP
% RUN NONNEGATIVE QUADRATIC PROGRAM ON MULTIPLE DIMENSIONS
%
% This script runs the NQP routine for various dimensions n. The script
% runs nonneg_quad_prog() and saves all output to memory, then writes 
% tables and plots to file later. It will save all console output to the 
% file nonneg_quad_prog_test.txt.
%
% coded by Kevin L. Keys (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% clean slate!
clear;
clc;

% record all screen output
diary on;
diary('nonneg_quad_prog_test.txt');

% timestamp for start of code
fprintf(['Start time: ', datestr(now)]);
fprintf('\n');

% set random seed
rng(12345);
fprintf('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n');
fprintf('-            Running nonnegative quadratic program!         -\n'); 
fprintf('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n');

% vector(s) of dimensions to test
N   = pow2(3:12);
NQP = zeros(length(N), 10);

for j = 1:length(N)
    
    n = N(j);
    
    fprintf('\n- - - Current dimension: %d - - - \n\n', n);
    
    % configure parameters for nonnegative quadratic programming
    temp = randn(n,n);
    
    % pad eigenvalues to ensure positive definiteness
    A = temp' * temp + eye(n,n);
    
    % no restrictions on b
    b = randn(n,1);
    
    % compute nonnegative quadratic program
    [a,c,d,e,f,g,h,k,l,p] = nonneg_quad_prog_test(A, b);
    NQP(j,:) = [a,c,d,e,f,g,h,k,l,p];
    
    % recouperate memory
    clear A b a c d e f g h k l p temp n;
    
end

% bookkeeping
fprintf(['End time: ', datestr(now)]);
fprintf('\n');
diary off;
save(['NQP_',datestr(now, 'ddmmmyyyy')]);

% now we can create tables and figures!
if 0 %% comment to turn on table/figure generation

% nonnegative quadratic program output
MM_iter   = NQP(:,1);
MM_time   = NQP(:,2);
MM_opt    = NQP(:,3);
CVX_iter  = NQP(:,4);
CVX_time  = NQP(:,5);
CVX_opt   = NQP(:,6);
M_time    = NQP(:,7);
M_opt     = NQP(:,8);
Y_time    = NQP(:,9);
Y_opt     = NQP(:,10);

% table with CPU times and optima
FID = fopen('/Users/kkeys/Desktop/MM_EP/mm_vs_cvx_NQP_table.tex', 'w');
fprintf(FID, '\\begin{table}[!ht]\n');
fprintf(FID, '\\begin{center}\n');
fprintf(FID, '\\begin{tabular}{rrrrrrrrr}\n');
fprintf(FID, '\\toprule\n');
fprintf(FID, '& \\multicolumn{4}{c}{CPU times} & \\multicolumn{4}{c}{Optima} \\\\ \n');
fprintf(FID, '\\cmidrule(r){2-5} \\cmidrule(r){6-9} \n');
fprintf(FID, '$d$ & MM & CV & MA & YA & MM & CV & MA & YA \\\\ \\hline \n');
for k=1:length(N)
    fprintf(FID, '%d & %3.3f & %3.3f & %3.3f & %3.3f & %3.4f & %3.4f & %3.4f & %3.4f\\\\ \n', ...
        N(k), MM_time(k), CVX_time(k), M_time(k), Y_time(k), MM_opt(k), CVX_opt(k), M_opt(k), Y_opt(k));
end
fprintf(FID, '\\bottomrule \n');
fprintf(FID, '\\end{tabular} \n');
fprintf(FID, '\\end{center} \n');
fprintf(FID, '\\caption{CPU time and optima for the nonnegative quadratic program. $d$ is for problem dimension. Abbreviations: MM for the proximal distance algorithm, CV for CVX, MA for MATLAB''s \\texttt{quadprog}, and YA for YALMIP.}\n');
fprintf(FID, '\\label{tab:mm_vs_cvx_NQP}\n');
fprintf(FID, '\\end{table}\n');
fclose(FID);

% plot with optima versus problem dimension
plot(log2(N), MM_opt, 'Color', 'b', 'Linewidth', 1);
axis([0, max(log2(N)) + 1, min(min([MM_opt, CVX_opt, M_opt, Y_opt])) - 2, max(max([MM_opt, CVX_opt, M_opt, Y_opt])) + 2]);
hold all;
plot(log2(N), CVX_opt, 'Color', 'r', 'Linewidth', 1);
plot(log2(N), M_opt, 'Color', 'k', 'Linewidth', 1);
plot(log2(N), Y_opt, 'Color', 'g', 'Linewidth', 1);
title('Optima versus Dimension for Nonnegative Quadratic Program ', 'Fontsize', 12);
xlabel('log_2(N)', 'Fontsize', 12);
ylabel('Optimal Values', 'Fontsize', 12);
legend('MM', 'CVX', 'MATLAB', 'YALMIP', 'Location', 'NorthWest');
saveas(gcf, '/Users/kkeys/Desktop/MM_EP/mm_vs_cvx_NQP_opt', 'png');
hold;

end % end switch for (not) printing tables and plot