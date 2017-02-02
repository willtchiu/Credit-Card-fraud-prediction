function lam = trainLambda(Xtrain, Xval, Ytrain, Yval, initial_theta)

lambda = [0.01, 0.1, 1, 10, 100, 1000, 3000 10000, 30000, 100000, 300000, 1000000];
options = optimset('GradObj', 'on', 'MaxIter', 500);
for lambda_train = lambda

    fprintf('----------------------------------------\n');
    fprintf('For lambda: %f\n', lambda_train);
    fprintf('----------------------------------------\n');

    [theta, cost] = ...
        fminunc(@(t)(costFunctionReg(t, Xtrain, Ytrain, lambda_train)), initial_theta, options);
    
    p = predict(theta, Xval);

    [prec, recall] = calcPR(p, Yval);


    fprintf('Precision: %f\n', prec);
    fprintf('Recall: %f\n\n', recall);

end

end