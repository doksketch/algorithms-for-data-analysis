import numpy as np
from sklearn.tree import DecisionTreeRegressor
from metrics import mean_squared_error
from metrics import bias

def gb_predict(X, trees_list, coef_list, eta):
    return np.array([sum([eta* coef * alg.predict([x])[0] for alg, coef in zip(trees_list, coef_list)]) for x in X])


def sgbm_fit((n_trees, max_depth, X_train, X_test, y_train, y_test, coefs, eta, batch_size):
   
    n = X_test.shape[1]
    
    n_batch = n // batch_size #количество элементов в 'порции'
    
    if n % batch_size != 0:
        n_batch += 1
        
    trees = []

    train_errors = []
    test_errors = []
    
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

        
        #формируем индексы объектов, которые должны попасть с батч
        b = np.random.randint(n_batch)
        start_ = batch_size*b 
        end_ = batch_size*(b+1) 

        #Выбор элементов в батч по индексам
        X_tmp = X_train[:, start_ : end_, :] 
        y_tmp = y_train[start_ : end_] 

        if len(trees) == 0:
            tree.fit(X_train, y_train)
                
            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))
        else:
            target = gb_predict(X_train, trees, coefs, eta)

            tree.fit(X_train, bias(y_train, target))
                
            train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))

        trees.append(tree)
        
    return trees, train_errors, test_errors