import numpy as np
from pyeda.boolalg.picosat import satisfy_one
from pyeda.boolalg.bfarray import exprvar
from pyeda.inter import *

import test_results

x = np.load("X_2term.npy")
y = np.load("Y_2term.npy")

features_N1 = []
features_P1 = []
features_Z1 = []
features_N2 = []
features_P2 = []
features_Z2 = []
clauses = []
for i in range(x.shape[1]):
    N1 = exprvar('N1_'+str(i+1))
    P1 = exprvar('P1_' + str(i+1))
    Z1 = exprvar('Z1_' + str(i+1))
    N2 = exprvar('N2_' + str(i+1))
    P2 = exprvar('P2_' + str(i+1))
    Z2 = exprvar('Z2_' + str(i+1))
    features_N1.append(N1)
    features_P1.append(P1)
    features_Z1.append(Z1)
    features_N2.append(N2)
    features_P2.append(P2)
    features_Z2.append(Z2)
    clauses.append(Or(N1, P1, Z1))
    clauses.append(Or(Not(N1), Not(P1)))
    clauses.append(Or(Not(N1), Not(Z1)))
    clauses.append(Or(Not(P1), Not(Z1)))
    clauses.append(Or(N2, P2, Z2))
    clauses.append(Or(Not(N2), Not(P2)))
    clauses.append(Or(Not(N2), Not(Z2)))
    clauses.append(Or(Not(P2), Not(Z2)))

A1 = exprvar('A1')
A2 = exprvar('A2')
clauses.append(Or(A1, A2))

for i in range(x.shape[0]):
    sample = x[i, :]
    clause1 = None
    clause2 = None
    if y[i] == 1:
        for j in range(len(sample)):
            feature = sample[j]
            if feature == 1:
                clauses.append(Or(Not(features_N1[j]), Not(A1)))
                clauses.append(Or(Not(features_N2[j]), Not(A2)))
                c1 = features_N1[j]
                c2 = features_N1[j]
            else:
                clauses.append(Or(Not(features_P1[j]), Not(A1)))
                clauses.append(Or(Not(features_P2[j]), Not(A2)))
                c1 = features_P1[j]
                c2 = features_P1[j]
            if clause1 is None:
                clause1 = c1
            else:
                clause1 = Or(clause1, c1)
            if clause2 is None:
                clause2 = c2
            else:
                clause2 = Or(clause2, c2)
        clauses.append(Or(clause1, A1))
        clauses.append(Or(clause2, A2))
    else:
        for j in range(len(sample)):
            feature = sample[j]
            if feature == 1:
                c1 = features_N1[j]
                c2 = features_N2[j]
            else:
                c1 = features_P1[j]
                c2 = features_P2[j]
            if clause1 is None:
                clause1 = c1
            else:
                clause1 = Or(clause1, c1)
            if clause2 is None:
                clause2 = c2
            else:
                clause2 = Or(clause2, c2)
        clauses.append(clause1)
        clauses.append(clause2)

sat = None
# cardinality constraint
k = 5
n = x.shape[1]
S = []
for i in range(n):
    sub_S = []
    for j in range(k):
        sub_S.append(exprvar('S_' + str(i+1)+str(j+1)))
    S.append(sub_S)

clauses.append(Or(Xor(features_Z1[0], features_Z2[0]), Not(S[0][0])))
clauses.append(Or(Not(Xor(features_Z1[0], features_Z2[0])), S[0][0]))
for j in range(1, k):
    clauses.append(Not(S[0][j]))
for i in range(1, n):
    clauses.append(Or(Not(Xor(features_Z1[i], features_Z2[i])), Not(S[i-1][k-1])))
for i in range(1, n-1):
    clauses.append(Or(Xor(features_Z1[i], features_Z2[i]), S[i-1][0], Not(S[i][0])))
    clauses.append(Or(Not(Xor(features_Z1[i], features_Z2[i])), S[i][0]))
    clauses.append(Or(Not(S[i-1][0]), S[i][0]))
for i in range(1, n-1):
    for j in range(1, k):
        clauses.append(Or(Xor(features_Z1[i], features_Z2[i]), S[i-1][j], Not(S[i][j])))
        clauses.append(Or(S[i-1][j-1], S[i-1][j], Not(S[i][j])))
        clauses.append(Or(Not(S[i-1][j]), S[i][j]))
        clauses.append(Or(Not(Xor(features_Z1[i], features_Z2[i])), Not(S[i-1][j-1]), S[i][j]))
clauses.append(Or(Xor(features_Z1[n-1], features_Z2[n-1]), S[n-2][k-1]))
clauses.append(Or(S[n-2][k-2], S[n-2][k-1]))

cnf = And(*clauses)
sats = cnf.satisfy_one()

if sats is None:
    print("no monomial exists")
    exit()
print(sats)

s = 'f ='
s1 = ' '
s2 = ' '
for k in sats:
    name = str(k)
    if sats[k] == 1 and name[0] != 'Z' and name[0] != 'A' and name[0] != 'S':
        if name[0] == 'P':
            if name[1] == '1':
                s1 = s1 + 'x' + name[3:]
            else:
                s2 = s2 + 'x' + name[3:]
        else:
            if name[1] == '1':
                s1 = s1 + '~x' + name[3:]
            else:
                s2 = s2 + '~x' + name[3:]
if len(s1) <= len(s2):
    s = s + s1 + ' +' + s2
else:
    s = s + s2 + ' +' + s1
print(s)
text_file = open("2term.txt", "w")
text_file.write(s)
text_file.close()

predict = test_results.check_2term(x, y, sats, features_N1, features_P1, features_N2, features_P2)
total = len(y)
assert (len(predict) == total)
results = sum(predict == y)
print("correct number is: %d" % results)
if results == total:
    print("The monomial is valid")
else:
    print("The monomial is not valid")

