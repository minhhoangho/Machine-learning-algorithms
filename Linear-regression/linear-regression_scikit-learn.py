from sklearn import  datasets, linear_model
import numpy as np
# mẫu số liệu  (x :Chiều cao, y: Cân nặng)
# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# weight (kg)
Y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

one = np.ones((X.shape[0], 1))
X_train = np.concatenate((one, X), axis=1)






#fit the model by linear regression

regression_model = linear_model.LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias

regression_model.fit(X_train, Y)

print( 'Solution found by scikit-learn  : ', regression_model.coef_ )
