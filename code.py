#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import matrix_rank

print ("Projet 1: calculs matriciels avec la librairie Numpy \n")
#vecteur y
def vecteur_y ():
	y = np.array([[1],[1],[1],[1],[1],[1],[1],[1]])
	return y

#=debug
#print ("Le vecteur y :  \n\n",vecteur_y())

#matrice identitée dans R⁸
def I_8 ():
	i = np.identity(8,dtype=int)
	return i

#==debug
#print ("La matrice identitée dans R⁸: \n\n",I_8())

#matrice E à construire
def matrice_E ():
	e = np.zeros((8,8),dtype=int)
	e[0,-1] = 1
	return e
#===debug
#print ("La matrice E : \n\n\n",matrice_E())

#matrice X : matrice d'observation
I_8 = np.identity(8,dtype=int)
matrice_E = np.zeros((8,8),dtype=int)
matrice_E[0,-1] = 1
matrix_x = np.add(I_8, matrice_E)
#====debug
#print ("La matrice X : \n\n\n", matrix_x)

print ("- Calcul de beta avec la formule du sujet")
#============utilisation des fonctions de la librairie numpy pour construire le beta
Transpose = matrix_x.transpose()
Produit1 = Transpose @ matrix_x
Inverse = np.linalg.inv(Produit1)
Produit2 = Inverse @ Transpose
vecteur_y = vecteur_y()

#beta devient donc:
beta = np.dot(Produit2,vecteur_y)
print ("β =  \n\n\n",beta)

#=====debug
#print ("Transpose : \n\n\n",Transpose)
#print ( "Produit 1 : \n\n\n",Produit1)
#print ("Produit 2 : \n\n\n",Produit2)
#print ("Vecteur y : \n\n\n",vecteur_y())

###programme principal
print ("                 8    \n")
print ("* Question 1:    ∑βi \n")
print ("                i=1 \n")
#calcul de sigma de 1 à 8 de beta i
print ("\n\n")
print ("La somme est de :",np.sum(vecteur_y))

#donner le rang de la matrice X^T X - I8
#calcul de la nouvelle matrice
#n = nouvelle matrice
n = np.subtract(Transpose @ matrix_x,I_8)
print ("X^T X - I8 = matrice n : \n\n",n)
print ("\n\n")

#calcul du rang de la matrice n
print ("** Question 2: \n")
n = matrix_rank(n)
print ("Le rang de la matrice n: \n",n)
