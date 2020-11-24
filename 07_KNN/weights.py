#функция для выбора необходимого метода расчёта весов
def get_weights(w, i, q, distance):
    if w == 'nn_weight':
        return 1/i #расчёт веса в зависимости от номера соседа
    elif w == 'n_dist_weiight':
        return q**distance #расчёт веса в зависимости от расстояния до соседа
    else:
        return 1 #расчёт веса не используем