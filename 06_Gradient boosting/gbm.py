#функция предсказания
def gb_predict(X, trees_list, coef_list, eta):
    # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
    # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании прибавляются с шагом eta
    return np.array([sum([eta* coef * alg.predict([x])[0] for alg, coef in zip(trees_list, coef_list)]) for x in X])

# функция обучения градиентного бустинга
def gb_fit(n_trees, max_depth, X_train, X_test, y_train, y_test, coefs, eta):

    #Записываем деревья в список
    trees = []

    #Записываем ошибки на каждой итерации для обучающей и тестовой выборок
    train_errors = []
    test_errors = []

    for i in range(n_trees):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    
    #инициализируем бустинг начальными алгоритмом, возвращающим ноль
    #поэтому первый алгоритм просто обучаем на выборке и добавляем в список
    if len(trees) == 0:
        #обучаем первое дерево
        tree.fit(X_train, y_train)

        train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
        test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))
    else:
        #получим ответы на текущей композиции
        target = gb_predict(X_train, trees, coefs, eta)

        #начиная со второго алгоритма обучаем на сдвиг
        tree.fit(X_train, bias(y_train, target))

        train_errors.append(mean_squared_error(y_train, gb_predict(X_train, trees, coefs, eta)))
        test_errors.append(mean_squared_error(y_test, gb_predict(X_test, trees, coefs, eta)))

    trees.append(tree)

    return trees, train_errors, test_errors