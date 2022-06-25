import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
diabetes=datasets.load_diabetes()
#['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module']
diabetes_x=diabetes.data[:,np.newaxis,2]
diabetes_x_train=diabetes_x[-30:]
diabetes_x_test=diabetes_x[:20]
diabetes_y_train=diabetes.target[-30:]
diabetes_y_test=diabetes.target[:20]
model=linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_x_test)
print("Mean square error is: ",mean_squared_error(diabetes_y_test,diabetes_y_predicted))
print("Weights: ",model.coef_)
print("Intercept is: ",model.intercept_)
plt.scatter(diabetes_x_test,diabetes_y_test)
plt.show()


