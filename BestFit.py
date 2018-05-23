# Programs best fit line (Linear regression) from scratch
# y = mx + b
# m = ((mean of all xs * mean of all ys) - (mean of xs*ys))/ ((means of xs to the power of 2) - (mean of all the xs to the power of 2))
# b = (mean of all ys) - m(mean of all xs)

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("fivethirtyeight")

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,7,8,9,7], dtype=np.float64)

# generates a scatter plot with the xs and ys as points
# plt.scatter(xs, ys)
# plt.show()

def best_fit_slope_and_intercept(xs, ys):
    # calculates slope
    m = ( ((mean(xs) * mean(ys)) - mean(xs*ys) ) /
            ( (mean(xs)*mean(xs)) - mean(xs**2)) )
    # calculates y-intercept
    b = mean(ys) - m*mean(xs)
    return m, b

m,b = best_fit_slope_and_intercept(xs, ys)

# creates regression line
regression_line = [(m*x)+b for x in xs]

# using regression line, predicts a point not already plotted
predict_x = 8
predict_y = (m*predict_x)+b


# generates scatter plot WITH the regression line
plt.scatter(xs, ys)
# highlights predicted point
plt.scatter(predict_x, predict_y, color="g")
plt.plot(xs, regression_line)
plt.show()
