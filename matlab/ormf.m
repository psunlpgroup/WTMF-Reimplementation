function [P, Q] = ormf(X, dim, lambda, w_m, alpha, maxiter)


t_begin = cputime;
%%% 1. initialization
[n_words, n_docs] = size(X);

fprintf('[ormf()]: n_words=%d n_docs=%d n_tokens=%d\n', n_words, n_docs, nnz(X));
fprintf('[ormf()]: dim=%d lambda=%f w_m=%f alpha=%f maxiter=%d\n', dim, lambda, w_m, alpha, maxiter);


% build index
i4d = cell(2,n_docs);
for j = 1:n_docs
    [i4d{1,j}, ~, i4d{2,j}] = find(X(:,j));
end

X_t = X';
i4w = cell(2,n_words);
for i = 1:n_words
    [i4w{1,i}, ~, i4w{2,i}] = find(X_t(:,i));
end


% initialize data structure
P = randn(dim, n_words);
Q = zeros(dim, n_docs);

clear X X_t;


%%% 2. gradient descent
for iter=1:maxiter
    %%% 2.1 compute Q
    fprintf('[ormf.m]: iteration=%d calculating Q...\n', iter);
    pptw = P*P'*w_m;
    for j = 1:n_docs
        pv = P(:,i4d{1,j});
        Q(:,j) = (pptw + pv*pv'*(1-w_m) + lambda*eye(dim))  \  (pv*i4d{2,j});
    end

        
    %%% 2.2 compute P
    fprintf('[ormf.m]: iteration=%d calculating P...\n', iter);
    qqtw = Q*Q'*w_m;
    for i = 1:n_words
        qv = Q(:,i4w{1,i});
        P(:,i) = (qqtw + qv*qv'*(1-w_m) + lambda*eye(dim))  \  (qv*i4w{2,i});
    end
    
    %%% 2.3 orthognal projection
    if alpha ~= 0
        P = P - alpha * (P*P'- diag(mean(diag(P*P'))*ones(dim,1)))*P;
    end
end


t_end = cputime;
fprintf('[ormf()]: used %f seconds\n', t_end-t_begin);

end

