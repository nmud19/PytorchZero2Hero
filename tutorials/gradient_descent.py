import numpy as np

# Equation : 2x+3
# General form => wx+b
# Objective : find w, b
x = np.array([1, 2, 3, 4, 5])
y = [(2 * X) + 3 for X in x]

# Setting up model
w = 0
b = 1

forward = lambda x, w, b: (x * w) + b
loss = lambda y, y_hat: ((y - y_hat) ** 2).mean()

# Gradient of MSE
# MSE = 1/N * ( w*x +b - y)**2
# d(mse)/dw = 1/n * 2( w*x + b - y) * x
# d(mse)/dw = 1/n * 2(yhat - y) * x

# d(mse)/db = 1/n * 2( w*x + b - y)
# d(mse)/db = 1/n * 2( yhat - y)

gradient_w = lambda x, y, y_hat: ((2 * x) * (y_hat - y)).mean()
gradient_b = lambda y, y_hat: (2 * (y_hat - y)).mean()

print("Prediction before training : ")
print(forward(x, w, b))

epochs = 100000
learning_rate = 0.001

for epoch in range(epochs+1):
    # forward pass
    y_hat = forward(x, w, b)

    # calculate loss
    loss_value = loss(y, y_hat)

    # calculate gradient + update weights
    dw = gradient_w(x, y, y_hat)
    db = gradient_b(y, y_hat)

    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)

    if epoch % 10000 == 0:
        print(f"epoch {epoch} : loss = {loss_value:.4f} , w = {w:.4f}, b = {b:.4f}")

print(f"Final predictions are : {y_hat}")
print(f"Expected values are   : {y}")
