def knn(X_train, y_train, X_test, k):
    answers = []

    for x in X_test:
        test_ditances = []

        for i in range(len(X_train)):
            # расчет расстояния от классифицируемого объекта до
            # объекта обучающей выборки
            distance = euclidean(x, X_train[i])

            # Записываем в список значение расстояния и ответа на объекте обучающей выборки
            test_distances.append((distance, y_train[i]))

        # создаем словарь со всеми возможными классами
        classes = {class_item: 0 for class_item in set(y_train)}

        # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
        for d in sorted(test_distances)[0:k]:
            classes[d[1]] += 1

        # Записываем в список ответов наиболее часто встречающийся класс
        answers.append(sorted(classes, key=classes.get)[-1])

        return answers