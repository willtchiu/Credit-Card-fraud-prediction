function [J, grad] = costFunctionReg(theta, X, y, lambda)

% Num of training examples
m = length(y);

theta_rest = theta(2:length(theta), 1);
[J, grad] = costFunction(theta, X, y);
J += (lambda/(2*m))*(sum(theta_rest.^2));

% Not regularizing the theta intercept
grad_zero = grad(1);
grad += (lambda/m)*theta;
grad(1) = grad_zero;

end