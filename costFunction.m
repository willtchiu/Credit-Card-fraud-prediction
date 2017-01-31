function [J, grad] = costFunction(theta, X, y)

m = length(y);

J = 0;
grad = zeros(size(theta));

J = (1/m)*sum((bsxfun(@times,-1.*y,log(sigmoid(X*theta)))).-(bsxfun(@times,1.-y,(log(1.-sigmoid(X*theta))))));
grad = (1/m)*X'*(sigmoid(X*theta)-y);

end