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

# cardinality constraint
k = 5
n = l
S = []
for i in range(n):
    sub_S = []
    for j in range(k):
        sub_S.append(exprvar('S_' + str(i+1)+str(j+1)))
    S.append(sub_S)

clauses.append(Or(Not(features_Z[0]), Not(S[0][0])))
clauses.append(Or(features_Z[0], S[0][0]))
for j in range(1, k):
    clauses.append(Not(S[0][j]))
for i in range(1, n):
    clauses.append(Or(features_Z[i], Not(S[i-1][k-1])))
for i in range(1, n-1):
    clauses.append(Or(Not(features_Z[i]), S[i-1][0], Not(S[i][0])))
    clauses.append(Or(features_Z[i], S[i][0]))
    clauses.append(Or(Not(S[i-1][0]), S[i][0]))
for i in range(1, n-1):
    for j in range(1, k):
        clauses.append(Or(Not(features_Z[i]), S[i-1][j], Not(S[i][j])))
        clauses.append(Or(S[i-1][j-1], S[i-1][j], Not(S[i][j])))
        clauses.append(Or(Not(S[i-1][j]), S[i][j]))
        clauses.append(Or(features_Z[i], Not(S[i-1][j-1]), S[i][j]))
clauses.append(Or(Not(features_Z[n-1]), S[n-2][k-1]))
clauses.append(Or(S[n-2][k-2], S[n-2][k-1]))

cnf = And(*clauses)
sats = cnf.satisfy_one()
if sats is None:
    print("no monomial exists")
    exit()

print(sats)
s = 'f = '
for k in sats:
    name = str(k)
    if sats[k] == 1 and name[0] != 'Z' and name[0] != 'S':
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
