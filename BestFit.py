# Programs best fit line (Linear regression) from scratch
# y = mx + b
# m = ((mean of all xs * mean of all ys) - (mean of xs*ys))/ ((means of xs to the power of 2) - (mean of all the xs to the power of 2))
# b = (mean of all ys) - m(mean of all xs)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use("fivethirtyeight")

# Testing data
# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,7,8,9,7], dtype=np.float64)

# generates a scatter plot with the xs and ys as points
# plt.scatter(xs, ys)
# plt.show()

# generates random dataset
def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation=="pos":
            val += step
        elif correlation and correlation=="neg":
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtape=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    # calculates slope
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys) ) /
            ( (mean(xs)*mean(xs)) - mean(xs**2)) )
    # calculates y-intercept
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_orig, ys_line):
    return sum(ys_line - ys_orig)**2

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/ squared_error_y_mean)

# calling create_dataset to generate a random dataset
xs, ys = create_dataset(40, 40, 2, correlation="pos")

m,b = best_fit_slope_and_intercept(xs, ys)

# creates regression line
regression_line = [(m*x)+b for x in xs]

# using regression line, predicts a point not already plotted
predict_x = 8
predict_y = (m*predict_x)+b


r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

# generates scatter plot WITH the regression line
plt.scatter(xs, ys)
# highlights predicted point
plt.scatter(predict_x, predict_y, color="g")
plt.plot(xs, regression_line)
plt.show()
