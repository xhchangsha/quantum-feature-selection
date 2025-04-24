function [W, alpha]= optimize_w_a(K, Y, max_iter, tol)
    % Input:
    % Y: Label matrix, dimension n x c
    % K: Kernel matrix set, dimension d x n x n (d is the number of kernel matrices)
    % max_iter: Maximum number of iterations
    % tol: Convergence threshold

    % Output:
    % W: Optimized matrix, dimension n x c
    % alpha: Optimized coefficients, dimension d x 1

    % Initialization
    Y=double(Y); 
    [n, c] = size(Y); % n is the number of samples, c is the number of classes
    d = size(K, 1);   % d is the number of kernel matrices, also the feature dimension

    % Initialize W as a semi-orthogonal matrix
    W = randn(n, c);
    [W, ~] = qr(W, 0); % Ensure W^T * W = I_c

    % Initialize alpha with uniform distribution
    alpha = ones(d, 1) / d;

    % Define H matrix
    H = eye(n) - (1/n) * ones(n, n);
    
    % Initialize objective function value
    F_pre = -100000;
    
    % Iterative optimization
    for iter = 1:max_iter
        % Fix alpha, update W
        G = zeros(n, n);
        for i = 1:d
            G = G + alpha(i) * squeeze(K(i, :, :));
        end
        
        A = G' * H * G;% Compute A = G^T * H * G
        
        r = max(eig(A)) + 1; % Initialize A_hat = r * I_n - A, ensuring A_hat is positive definite, where r is A's maximum eigenvalue plus 1
        A_hat = r * eye(n) - A;
        
        B = G' * H * Y; % B = G^T * H * Y
         
        M = 2 * A_hat * W + 2 * B;
        [U, ~, V] = svd(M, 'econ'); % SVD decomposition
        W = U * V'; % Update W

        % Fix W, update alpha
        % onstruct quadratic programming parameters, compute quadratic coefficient matrix Q and linear coefficient vector f
        Q = zeros(d, d);
        f = zeros(d, 1);
        for i = 1:d
            for j = 1:d
                Q(i, j) = trace((squeeze(K(i, :, :)) * W)'  * H * squeeze(K(j, :, :)) * W);
            end
            f(i) = -2 *trace(Y' * H * squeeze(K(i, :, :)) * W );
        end
        % Equality constraint: ∑α_i = 1
        Aeq = ones(1, d); % Row vector of all ones
        beq = 1;

        % Inequality constraint: α_i >= 0
        lb = zeros(d, 1); % 下界为 0
        
        % Call quadprog to solve
        alpha = quadprog(Q, f, [], [], Aeq, beq, lb);%, [], [],options);
        
        % Compute objective function value
        Temp = zeros(n, n);
        for i = 1:d
            Temp = Temp + alpha(i) * squeeze(K(i, :, :));
        end      
        
        F =norm(H*Y-H*G*W, 'fro')^2;
        
        % Check convergence condition
        if iter > 1
            if abs(F - F_pre) < tol
                fprintf('收敛于第 %d 次迭代\n', iter);
                break;
            end
        end    
       F_pre = F; % Save current F
    end;
    




