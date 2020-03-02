import numpy as np


def check_monomial(x, y, sats, features_N, features_P):
    total = len(y)
    predict = []
    for i in range(total):
        sample = x[i, :]
        for j in range(len(sample)):
            feature = sample[j]
            if feature == 1 and sats[features_N[j]] == 1:
                predict.append(0)
                break
            elif feature == 0 and sats[features_P[j]] == 1:
                predict.append(0)
                break
            if j == len(sample) - 1:
                predict.append(1)
    predict = np.asarray(predict)
    return predict


def check_2term(x, y, sats, features_N1, features_P1, features_N2, features_P2):
    predict_1 = check_monomial(x, y, sats, features_N1, features_P1)
    predict_2 = check_monomial(x, y, sats, features_N2, features_P2)
    predict = predict_1 + predict_2
    predict[predict > 1] = 1
    return predict
