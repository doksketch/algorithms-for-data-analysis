import numpy as np


#Подсчёт логистической функции потерь с проверкой на ненулевое выражение под логарифмом
def calc_logloss(y, y_pred):
    tol = 1e-5
    y_pred = y_pred.copy()
    y_pred = np.clip(y_pred, a_min = tol, a_max=1-tol)
    
    err = - np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return err

#сигмоида
def sigmoid(z):
    res = 1 / (1 + np.exp(-z))
    return res


#логит с регуляризацией
def eval_logit(X, y, iterations=None, alpha=None, lambda_1=0, lambda_2=0): 
    
    np.random.seed(42)
    W = np.random.randn(X.shape[0])
    n = X.shape[1]
    
    for i in range(1, iterations+1):
        z = np.dot(W, X)
        y_pred = sigmoid(z)
        err = calc_logloss(y, y_pred)
        
        W -= alpha * ((1/n * np.dot((y_pred - y), X.T)) +  2 * lambda_2 * W + (lambda_1 * W)/(np.abs(lambda_1)))
    
        if i % (iterations // 3) == 0:
            print(i, W, err)
    
    return W


#функция, расчитывающая вероятность отнесения к классу 1
def calc_pred_proba(W, X):
    y_pred_proba = 1/(1 + np.exp(-(np.dot(X,W))))
    return y_pred_proba


# Функция, рассчитывающая вероятность отнесения к классу 0 или 1
def calc_pred(W, X, treshhold):
    m = y_pred_proba.shape[0]
    y_pred = np.zeros((1, m))
    y_pred = np.array([], dtype='int32')
    
    y_pred_proba = calc_pred_proba(W, X)
    
    for i in y_pred_proba:
        
        if i > treshhold:
            y_pred = np.append(y_pred, 1)
        else:
            y_pred = np.append(y_pred, 0)
    
    return y_pred
