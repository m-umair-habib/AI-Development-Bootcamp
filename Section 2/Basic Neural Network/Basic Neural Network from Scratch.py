# ----- Developed By: Muhammad Umair Habib -----

# Import Libraries
import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error loss function
def mean_square_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Basic Neural Network Class
class BasicNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weight_input_hidden = np.random.randn(input_size, hidden_size)
        self.weight_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(1, hidden_size)
        self.bias_output = np.random.randn(1, output_size)

    # Forward Pass
    def forward(self, x):
        self.hidden_input = np.dot(x, self.weight_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weight_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output
    
    # Backward Pass and Weights Update
    def backward(self, x, y, output, learning_rate):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = np.dot(output_delta, self.weight_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        self.weight_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis = 0, keepdims=True) * learning_rate
        self.weight_input_hidden += np.dot(x.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # Train the neural network
    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward Pass
            output = self.forward(x)

            self.backward(x, y, output, learning_rate)

            if epoch % 100 == 0:
                loss = mean_square_error(y, output)
                print(f"Epoch: {epoch}, Loss: {loss}")


# XOR dataset
x = np.array([[0, 0], [0, 1], [1, 0], [1,1]])
y = np.array([[0], [1], [1], [0]])

nn = BasicNeuralNetwork(input_size=2, hidden_size=2, output_size=1)

nn.train(x, y, epochs=10000, learning_rate=0.1)

print(f"\n Test the trained neural network: ")
for i in range(len(x)):
    print(f"Input: {x[i]}, Predicted Output: {nn.forward(x[i])}, Actual Output: {y[i]}")