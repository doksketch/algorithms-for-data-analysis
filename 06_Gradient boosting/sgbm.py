import numpy as np
from sklearn.tree import DecisionTreeRegressor
from metrics import mean_squared_error
from metrics import bias

def sgbm_predict(X, trees_list, coef_list, eta):
    return np.array([sum([eta* coef * alg.predict([x])[0] for alg, coef in zip(trees_list, coef_list)]) for x in X])


def sgbm_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, coefs, eta, batch_size):
    
    trees = []
    
    train_errors = []
    test_errors = []
    
    n = X_train.shape[0]
    n_batch = int(n // batch_size) #количество элементов в батче

    batch = 0
    
    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)

        # границы батча
        batch = np.random.randint(batch_size)      
        start_ = n_batch*batch
        end_ = n_batch*(batch+1)
        
        #батч
        X_train_tmp = X_train[start_ : end_, :]
        y_train_tmp = y_train[start_ : end_]
        
        if len(trees) == 0:
            tree.fit(X_train_tmp, y_train_tmp)
            
            train_errors.append(mean_squared_error(y_train, sgbm_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, sgbm_predict(X_test, trees, coefs, eta)))
        else:
            target = sgbm_predict(X_train_tmp, trees, coefs, eta)
            
            tree.fit(X_train_tmp, bias(y_train_tmp, target))
            
            train_errors.append(mean_squared_error(y_train, sgbm_predict(X_train, trees, coefs, eta)))
            test_errors.append(mean_squared_error(y_test, sgbm_predict(X_test, trees, coefs, eta)))

        trees.append(tree)
        
    return trees, train_errors, test_errors