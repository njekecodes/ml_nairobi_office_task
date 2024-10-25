import numpy as nump


# MSE function
def mean_squarred_error(y_true, y_pred):
    return nump.mean((y_true - y_pred) ** 2)


def gradient_descent(x, y, m, c, learn_rate, epochs):
    n = len(y)
    for epoch in range(epochs):
        y_pred = m * x + c

        # Gradients
        dm = (-2 / n) * sum(x * (y - y_pred))
        dc = (-2 / n) * sum(y - y_pred)

        # update m and c
        m = m - learn_rate * dm
        c = c - learn_rate * dc

        # Calculate and print MSE for each epoch
        mse = mean_squarred_error(y, y_pred)
        print(f"Epoch {epoch + 1}: MSE = {mse}")

    return m, c
