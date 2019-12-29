function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h_theta = (theta.'*X.');
h_theta = sigmoid(h_theta);
theta_regularized = theta;
theta_regularized(1) = [];


J = (1/m)*(-y'*log(h_theta') - (1-y)'*log (1 - h_theta)') + (lambda/(2*m))*(theta_regularized'*theta_regularized);
%J = sum(J); J = sum(J);



grad = zeros(size(theta));
grad(1) = (1/m)*(h_theta - y')*X(:,1);

for a = 2:size(theta)
grad(a) = (1/m)*(h_theta - y')*X(:,a) + (lambda/m)*theta(a);
end


% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
