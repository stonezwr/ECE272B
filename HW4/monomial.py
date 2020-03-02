import numpy as np
from pyeda.boolalg.picosat import satisfy_one
from pyeda.boolalg.bfarray import exprvar
from pyeda.inter import *

import test_results

x = np.load("X_monomial.npy") * 1
y = np.load("Y_monomial.npy") * 1
l = x.shape[1]

features_N = []
features_P = []
features_Z = []
clauses = []
for i in range(l):
    N = exprvar('N' + str(i+1))
    P = exprvar('P' + str(i+1))
    Z = exprvar('Z' + str(i+1))
    features_N.append(N)
    features_P.append(P)
    features_Z.append(Z)
    # one-hot
    clauses.append(Or(N, P, Z))
    clauses.append(Or(Not(N), Not(P)))
    clauses.append(Or(Not(N), Not(Z)))
    clauses.append(Or(Not(P), Not(Z)))

for i in range(x.shape[0]):
    sample = x[i, :]
    clause = None
    if y[i]:
        for j in range(l):
            feature = sample[j]
            if feature == 1:
                clauses.append(Not(features_N[j]))
            else:
                clauses.append(Not(features_P[j]))
    else:
        for j in range(l):
            feature = sample[j]
            if feature == 1:
                c = features_N[j]
            else:
                c = features_P[j]
            if clause is None:
                clause = c
            else:
                clause = Or(clause, c)
        clauses.append(clause)

cnf = And(*clauses)
sats = cnf.satisfy_count()
print(sats)
exit()
sats = None
found = 0
for l1 in range(l):
    if found == 1:
        break
    if l1 + 1 > l:
        break
    for l2 in range(l1+1, l):
        if found == 1:
            break
        if l2 + 1 > l:
            break
        for l3 in range(l2 + 1, l):
            if found == 1:
                break
            if l3 + 1 > l:
                break
            for l4 in range(l3 + 1, l):
                if found == 1:
                    break
                if l4 + 1 > l:
                    break
                for l5 in range(l4 + 1, l):
                    if found == 1:
                        break
                    z = []
                    for i in range(l):
                        if i != l1 and i != l2 and i != l3 and i != l4 and i != l5:
                            z.append(features_Z[i])
                    final_clauses = clauses + z
                    cnf = And(*final_clauses)
                    sats = cnf.satisfy_one()
                    if sats is not None:
                        found = 1

print(sats)
s = 'f = '
for k in sats:
    name = str(k)
    if sats[k] == 1 and name[0] != 'Z':
        if name[0] == 'P':
            s = s + 'x' + name[1:]
        else:
            s = s + '~x' + name[1:]
print(s)
text_file = open("monomial.txt", "w")
text_file.write(s)
text_file.close()

predict = test_results.check_monomial(x, y, sats, features_N, features_P)
total = len(y)
assert (len(predict) == total)
results = sum(predict == y)
print("correct number is: %d" % results)
if results == total:
    print("The monomial is valid")
else:
    print("The monomial is not valid")
