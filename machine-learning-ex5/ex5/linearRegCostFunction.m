function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = (1/(2*m))*sum(((X*theta) - y).^2) + (lambda/(2*m))*(sum(theta.^2) - theta(1)^2);

grad = zeros(size(theta));


for j = 0:max(size(theta))-1  % theta would have have to be passed as column vector
    
    if j == 0
        grad(j+1) = (1/m)*sum(((X*theta) - y).*X(:,j+1))
    else
        grad(j+1) = (1/m)*sum(((X*theta) - y).*X(:,j+1)) + (lambda/m)*theta(j+1)
    end
end

        
    


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
