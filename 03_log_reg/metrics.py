import numpy as np

#Матрица ошибок 1 и 2 рода
def matrix_errors(y, y_pred):
    spam = list(zip(y, y_pred))
     
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i, j in enumerate(spam):
        if spam[i][0] == 1 and spam[i][1] == 1: #хорошего человека назвали хорошим
            TP += 1
        elif spam[i][0] < spam[i][1]: #плохого человека назвали хорошим
            FP += 1
        elif spam[i][0] < spam[i][1]: #хорошего человека назвали плохим
            FN += 1
        else:
            TN += 1 #плохого человека назвали плохим

    errors = np.array([TP, TN, FP, FN], dtype='int32')
    
    return errors

#Accuracy
def calc_accuracy(errors):
    acc = (np.sum([errors[0],errors[1]]))/(np.sum([errors]))
    return acc

#Precision
def calc_precision(errors):
    prec = errors[0]/(np.sum([errors[0], errors[2]]))   
    return prec

#Recall
def calc_recall(errors):
    rec = errors[0]/(np.sum([errors[0], errors[3]]))    #TP+FN
    return rec

#F1 score
def f1_score(prec, rec):
    f1 = (2*prec*rec)/(prec+rec)
    return f1