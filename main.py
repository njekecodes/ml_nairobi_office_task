import pandas as pd
import numpy as nump
from functions import mean_squarred_error, gradient_descent
from plot import plot_results

#Load the data
data = pd.read_csv('data/Nairobi Office Price Ex.csv')
x = data['SIZE'].values
y = data['PRICE'].values

m, c = nump.random.rand(), nump.random.rand()
learn_rate = 0.0001
epochs = 10

m, c = gradient_descent(x, y, m, c, learn_rate, epochs)

y_pred = m * x + c

plot_results(x, y, y_pred)

office_size = 100
predict_price = m * office_size + c
print(f"Predicted office price for 100 sq. ft: {predict_price}")
