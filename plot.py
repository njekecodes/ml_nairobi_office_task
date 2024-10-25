import matplotlib.pyplot as plt

def plot_results (x, y, y_pred):
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_pred, color='red', label='Line of Best Fit')
    plt.xlabel("Office Size (sq. ft)")
    plt.legend()
    plt.show()