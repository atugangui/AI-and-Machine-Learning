import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


regr = linear_model.LinearRegression()


''' This function generates a random 4th degree polynomial function, returns 4 arrays of numbers (each containing n 
numbers), and plots each data set in scattered form. Two arrays (of x and y values) form the training data, which 
follows the polynomial function but has added noise to the y values. The other two arrays (of x and y values) form the 
testing data, which are just random points along the original polynomial function.'''


def gen_data(n):
    coeff = np.array([np.random.uniform(-0.5, 0.5, 5)])
    powers = np.array([4, 3, 2, 1, 0])

    X_train = np.random.uniform(-2.0, 2.0, n)
    y_train = np.array([])

    X_test = np.linspace(-2.0, 2.0, num=n)
    y_test = np.array([])

    for z, j in zip(X_train, X_test):
        temp_train = np.array([np.power(z, powers)])
        temp_test = np.array([np.power(j, powers)])
        for i, o in zip(temp_train, temp_test):
            y_train = np.append(y_train, np.sum(np.array([i*coeff])))
            y_test = np.append(y_test, np.sum(np.array([o*coeff])))

    noise = np.random.normal(0,1,n)
    y_train = y_train + noise

    if n == 1000:
        size = 3
    else:
        size = 10

    plt.scatter(X_test, y_test, c="green", s=size, alpha=0.5, label='Testing Data')
    plt.scatter(X_train, y_train, c="blue", s=size, alpha=0.5, label='Training Data')

    return X_train, y_train, X_test, y_test


''' This function is very similar to the gen_data function, except it does not plot the data sets.'''


def gen_only_data(n):
    coeff = np.array([np.random.uniform(-0.5, 0.5, 5)])
    powers = np.array([4, 3, 2, 1, 0])

    X_train = np.random.uniform(-2.0, 2.0, n)
    y_train = np.array([])

    X_test = np.linspace(-2.0, 2.0, num=n)
    y_test = np.array([])

    for z, j in zip(X_train, X_test):
        temp_train = np.array([np.power(z, powers)])
        temp_test = np.array([np.power(j, powers)])
        for i, o in zip(temp_train, temp_test):
            y_train = np.append(y_train, np.sum(np.array([i*coeff])))
            y_test = np.append(y_test, np.sum(np.array([o*coeff])))

    noise = np.random.normal(0, 1, n)
    y_train = y_train + noise

    return X_train, y_train, X_test, y_test


''' This function plots the learning curves for each degree (0-20) by creating a polynomial model of that degree, 
fitting a random data set, and calculating the MSE. It does this 100 times for each degree, and then records the average
MSE value for that degree. It then plots the average MSE for both the testing data and the training data against the 
 degree of the model. It does this for each of the three conditions (10, 100, and 1000 data points).'''


def learning_curves(n, rep):
    train_err = []
    val_err = []
    degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for i in range(21):
        train_errors, val_errors = [], []
        for m in range(rep):
            X_train, y_train, X_val, y_val = gen_only_data(n)
            model = make_pipeline(PolynomialFeatures(degrees[i]), regr)
            model.fit(X_train[:, np.newaxis], y_train)
            train_predict = model.predict(X_train[:, np.newaxis])
            val_predict = model.predict(X_val[:, np.newaxis])
            train_errors.append(mean_squared_error(train_predict, y_train))
            val_errors.append(mean_squared_error(val_predict, y_val))
        mean_train_error = sum(train_errors)/len(train_errors)
        mean_val_error = sum(val_errors)/len(val_errors)
        train_err.append(mean_train_error)
        val_err.append(mean_val_error)
    plt.ylim((0, 10))
    plt.xticks(np.arange(0, 21, 1.0))
    plt.plot(degrees, train_err, "r-+", linewidth=2, label="Training")
    plt.plot(degrees, val_err, "b-+", linewidth=3, label="Testing")
    plt.title('{} Data Points'.format(n))
    plt.xlabel('Degree')
    plt.ylabel('Average MSE Over 100 Runs')
    plt.legend()
    plt.show()


''' This function creates three models of polynomial degrees 1, 2, and 20 and fits them to a training data set. 
 It then plots these models for data sets of size 10, 100, and 1000.'''


def poly_model():
    num = [10, 100, 1000]
    for m in range(3):
        degrees = [1,2,20]
        color = ['orange', 'k', 'm']
        X_train, y_train, X_test, y_test = gen_data(num[m])
        for i in range(3):
            model = make_pipeline(PolynomialFeatures(degrees[i]), regr)
            model.fit(X_train[:, np.newaxis], y_train)
            poly_predict = model.predict(X_test[:, np.newaxis])
            plt.ylim(-10, 10)
            plt.plot(X_test, poly_predict, c=color[i], label='Degree {}'.format(degrees[i]))
            plt.title('{} Data Points'.format(num[m]))
            plt.xlabel('X Values')
            plt.ylabel('Y Values')
            plt.legend()
        plt.show()


poly_model()

learning_curves(10, 100)
learning_curves(100, 100)
learning_curves(1000, 100)
