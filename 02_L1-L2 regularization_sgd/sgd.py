import numpy as np

# реализуем функцию, определяющую среднеквадратичную ошибку
def calc_mse(y, y_pred):
    err = np.mean((y - y_pred)**2)
    return err

#стохастический градиентный спуск с регуляризацией
def eval_sgd_model_reg(X, y, iterations=None, batch_size=None, alpha=None, lambda_1=0, lambda_2=0):
    W = np.random.randn(X.shape[0])
    n = X.shape[1]
    n_batch = n // batch_size #количество элементов 
    if n % batch_size != 0: #если количество элементов нечётное
        n_batch += 1
    
    errors = []
    num_iterations = []
    
    for i in range(1, iterations+1):
        for b in range(n_batch):
            start_ = batch_size*b #вычисляем индексы объектов, которые должны попасть
            end_ = batch_size*(b+1) #вычисляем индексы объектов, которые должны попасть

            # print(b, n_batch, start_, end_)

            X_tmp = X[:, start_ : end_] #выбор элемента
            y_tmp = y[start_ : end_] #выбор элемента
            y_pred_tmp = np.dot(W, X_tmp)
            err = calc_mse(y_tmp, y_pred_tmp)
            W -= (alpha * (1/n * 2 * np.dot((y_pred_tmp - y_tmp), X_tmp.T)) +  2 * lambda_2 * W + (lambda_1 * W)/(np.abs(lambda_1)))

        if i % (iterations / 5) == 0:
            errors.append(err)
            num_iterations.append(i)
    
    return num_iterations, errors