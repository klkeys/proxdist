% run_NQP
% RUN PIECEWISE LINEAR PROGRAM OVER BINARY SET ON MULTIPLE DIMENSIONS
%
% This script runs the piecewiselin_over_binary() routine for various 
% problem dimensions n. The script saves all function output to memory, 
% then writes tables and plots to file later. It will save all console 
% output to the file piecewiselin_over_binary_test.txt.
%
% coded by Kevin L. Keys (2014)
% klkeys@ucla.edu
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

% clean slate!
clear;
clc;

% record all screen output
diary on;
diary('piecewiselin_over_binary_test.txt');

% timestamp for start of code
fprintf(['Start time: ', datestr(now)]);
fprintf('\n');

% set random seed
rng(12345);

fprintf('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n');
fprintf('-          Running piecewise linear binary program!         -\n'); 
fprintf('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n');

% vector(s) of dimensions to test
N = pow2(1:12);

% store output
% must store for length(N) dimensions tested
PWLoBox = zeros(length(N), 10);

for j = 1:length(N)
    
    n = N(j);
    
    fprintf('\n- - - Current dimension: %d - - - \n\n', n);
% 
%     % setup parameters for graph cut problem
    temp = randn(n,n);
    A    = abs(triu(temp,1) + triu(temp, 1)');
    b    = n*randn(n,1);

    % compute binary piecewise linear objective
    fprintf('\n- - - Piecewise linear over binary - - -\n');
    [a,c,d,e,f,g,h,k,l,m] = piecewiselin_over_binary(A, b);
    PWLoBox(j,:) = [a,c,d,e,f,g,h,k,l,m];
    
    % shore up some memory for next step
    clear A b temp a c d e f g h k l m;
   
end

% final memory wipe
clear j N;

% bookkeeping
fprintf(['End time: ', datestr(now)]);
fprintf('\n');
diary off;
save(['PWLoBox_',datestr(now, 'ddmmmyyyy')]);

% now we can create tables and figures!
if 0 %% comment to turn on table/figure generation

% binary piecewise linear program output
MM_iter   = PWLoBox(:,1);
MM_time   = PWLoBox(:,2);
MM_opt    = PWLoBox(:,3);
CVX_iter  = PWLoBox(:,4);
CVX_time  = PWLoBox(:,5);
CVX_opt   = PWLoBox(:,6);
Y_time    = PWLoBox(:,7);
Y_opt     = PWLoBox(:,8);
diffs     = PWLoBox(:,9);
convd     = PWLoBox(:,10);

% table with both CPU times and optima
FID = fopen('/Users/kkeys/Desktop/MM_EP/mm_vs_cvx_PWLoBox_table.tex', 'w');
fprintf(FID, '\\begin{table}[!ht]\n');
fprintf(FID, '\\begin{center}\n');
fprintf(FID, '\\begin{tabular}{rrrr}\n');
fprintf(FID, '\\toprule\n');
fprintf(FID, '& \\multicolumn{2}{c}{CPU times} \\\\ \n');
fprintf(FID, '\\cmidrule(r){2-3} \n');
fprintf(FID, 'Dimension & \\#1 & \\#2 & \\#3 \\\\ \\hline \n');
for k=1:length(N)
    fprintf(FID, '%d & %3.3f & %3.3f & %d \\\\ \n', N(k), MM_time(k), CVX_time(k), diffs(k));
end
fprintf(FID, '\\bottomrule \n');
fprintf(FID, '\\end{tabular} \n');
fprintf(FID, '\\end{center} \n');
fprintf(FID, '\\caption{CPU time and optima for binary piecewise linear function. Abbreviations: $d$ for dimension, \\#1 for proximal algorithm, \\#2 for CVX, \\#3 for the number of components that vary between the solutions from \\#1 and \\#2.}\n');
fprintf(FID, '\\label{tab:mm_vs_cvx_PWLoBox}\n');
fprintf(FID, '\\end{table}\n');
fclose(FID);

% plot with optima versus dimension
plot(log2(N), MM_opt, 'Color', 'b', 'Linewidth', 1);
axis([0, max(log2(N)) + 1, min(min([MM_opt, CVX_opt])) - 2, max(max([MM_opt, CVX_opt])) + 2]);
hold all;
plot(log2(N), CVX_opt, 'Color', 'r', 'Linewidth', 1);
title('Optima versus Dimension for Binary PIecewise Linear Objective ', 'Fontsize', 12);
xlabel('log_2(N)', 'Fontsize', 12);
ylabel('Optimal Values', 'Fontsize', 12);
legend('MM', 'CVX', 'Location', 'NorthWest');
saveas(gcf, '/Users/kkeys/Desktop/MM_EP/mm_vs_cvx_PWLoBox_opt', 'png');
hold;

end % end switch for (not) creating table plot