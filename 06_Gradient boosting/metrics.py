#Расчёт mse
def mean_squared_error(y_real, prediction):
    return (sum((y_real - prediction)**2)) / len(y_real)

#Квадратичная функция потерь
def bias(y, z):
    return (y - z)