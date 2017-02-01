function p = predict(theta, X)

m = size(X, 1);

p = zeros(m, 1);

p = floor(sigmoid(X*theta)/.5);
p = p ~= 0;

end 