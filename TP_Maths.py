# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:25:08 2021

@author: Hansali haitam
"""
#Exercice 1:
def dimensions(A):
    i = len(A[0])
    j = len(A)
    return j,i
    
def affiche(A):
    for ligne in A :
        print(ligne)
        
def matriceNulle(n,p):
    return [[0 for j in range(p)] for i in range(n)]

def matriceUnite(n):
    mat = matriceNulle(n,n)
    for i in range(len(mat)):
        mat[i][i] = 1
    return mat
    
def transpose(A):
    mat = [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]
    return mat
    
def estTriangulaireSup(A):
    for i in range(1,len(A)):
        for j in range(i):
            if A[i][j] != 0 :
                return False
    return True

#Exercice 2 :
def sommeMatrice(A, B):
    n , p  = dimensions(A)
    q , r = dimensions(B)
    assert (n,p) == (q,r)
    C = matriceNulle(n,p)
    for i in range(len(A)):
        for j in range(len(A[0])):
            C [i][j] = A[i][j] + B[i][j]
    return C

def multScalaire(A,l):
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] *= l
    return A

def produitMatrice(A,B):
    C = matriceNulle(len(A),len(A[0]))
    print(C)
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C

#Exercice 3:
def blocs(A, a, b):
    n,p = dimensions(A)
    assert n == p
    B = matriceNulle(n*a, n*b)
    for j in range(len(B)):
        for k in range(len(B[0])):
            B[j][k] = A[j%n][k%n]
    return B

#Exercice 4:
def exponaif(A, n):
    if n == 0 :
        return matriceUnite(len(A))
    return produitMatrice(A, exponaif(A, n-1))

import numpy as np

A = np.array([[0, 1],[1, 1]], dtype = np.int64)
print(np.linalg.matrix_power(A, 45))

#Exercice 5:
def expoRapRec(x, n):
    if n == 0:
        return 1
    return expoRapRec(x**2, n//2) if n %2 == 0 else x*expoRapRec(x**2, (n-1)//2)
        
def expoRapMatrice(M, n):
    return [[expoRapRec(M[j][i], n) for i in range(len(M[0]))] for j in range(len(M))]

#xercice 6:
A = [[0, 1], [1, 1]]

print(expoRapMatrice(A, 50)) 
X = [0, 1]
for _ in range(100):
    X = np.dot(A,X)
    print(X) 
    