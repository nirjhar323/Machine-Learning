function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
z_size = size(z);
g = zeros(size(z));

for a = 1:z_size(1)
for b = 1:z_size(2)
g(a,b) = 1/(1+exp(-z(a,b)));
end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end
