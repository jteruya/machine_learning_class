function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Calculate Layers
hidden = sigmoid(([ones(m,1), X] * transpose(Theta1)));
output = sigmoid(([ones(m,1), hidden] * transpose(Theta2)));

% Create y value matrix
y_value = zeros(m,num_labels);

% Populate y value matrix with y labels vector
for i = 1:m
	y_value(i,y(i)) = 1;
endfor

% Calculate Element wise
element_cost = (-1/m) * ((y_value .* log(output)) + ((1 - y_value) .* log(1 - output)));

Theta1_reg_sq = Theta1(:,2:(input_layer_size+1)).^2;
Theta2_reg_sq = Theta2(:,2:(hidden_layer_size+1)).^2;

reg_cos_term = (sum(Theta1_reg_sq(:)) + sum(Theta2_reg_sq(:))) * lambda/(2*m);

% Cost
J = sum(element_cost(:)) + reg_cos_term;


% Back Propagation

for i = 1:m

	% Output Level Delta
	delta_output_i = transpose(output(i,:) - y_value(i,:));
	
	% Hidden Level Delta
	delta_hidden_i = transpose(Theta2) * delta_output_i .* ([1;transpose(hidden(i,:))] .* (1 - [1;transpose(hidden(i,:))]));

	% Accumulate Deltas
	Theta2_grad = Theta2_grad + delta_output_i * [1,hidden(i,:)];
	Theta1_grad = Theta1_grad + delta_hidden_i(2:end,:) * [1,X(i,:)];

endfor

Theta2_grad = 1/m * Theta2_grad + lambda/m * [zeros(num_labels,1),Theta2(:,2:end)];
Theta1_grad = 1/m * Theta1_grad + lambda/m * [zeros(hidden_layer_size,1),Theta1(:,2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
