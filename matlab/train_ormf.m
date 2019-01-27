function [] = train_ormf(data_file, model_file, dim, lambda, w_m, alpha, n_iters)

text_train = load(data_file);
text_train = spconvert(text_train);

[P] = ormf(text_train, dim, lambda, w_m, alpha, n_iters);
save(model_file, 'P', 'dim', 'lambda', 'w_m', 'alpha', 'n_iters');
exit;
end
