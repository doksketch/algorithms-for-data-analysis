import numpy as np

# реализуем функцию, определяющую среднеквадратичную ошибку
def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

#градиентный спуск
def eval_gd_model(X, y, iterations, alpha=1e-4):
    W = np.random.randn(X.shape[1])
    n = X.shape[0]
    errors = []
    num_iterations = []
    
    for i in range(1, iterations+1):
        y_pred = np.dot(X, W)
        err = calc_mse(y, y_pred)
        W -= (alpha * (1/n * 2 * np.dot(X.T, (y_pred - y))))
        
        if i % (iterations / 5) == 0:
            errors.append(err)
            num_iterations.append(i)
    
    return num_iterations, errors 