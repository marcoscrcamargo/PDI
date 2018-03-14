# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 - 1/2018
# Trabalho 1 - Gerador de imagens

# imports
import numpy as np
import random
import math

# Funções para geração da imagem da cena.

# Função 1 - x + y.
def f_one(x, y, Q):
	return (x + y)

# Função 2 - |sin(x/Q) + sin(y/Q)|.
def f_two(x, y, Q):
	return abs( math.sin(x/Q) + math.sin(y/Q) )

# Função 3 - (x/Q) - sqrt(y/Q).
def f_three(x, y, Q):
	return (x/Q) - math.sqrt(y/Q)

# Função 4 - valor aleatório continuo entre 0 e 1.
def f_four(x, y, Q):
	return random.random()

# Função 5 - caminho aleatório.
def f_five(x, y, Q):
	return 0

# Função que aplica uma função em todos os elementos da matriz.
def apply(m, f, Q):
	for (x,y), value in np.ndenumerate(m):
		m[x,y] = f(x, y, Q)
	
	return m

def main():
	# Recebendo parametros.
	# Nome do arquivo.
	fname = input()
	# Tamanho da lateral da cena (C x C).
	C = int(input())
	# Função a ser utilizada (1, 2, 3, 4, 5).
	f = int(input())
	# Parametro Q para calculo nas funções. 
	Q = int(input())
	# Tamanho lateral da imagem digital (N x N).
	N = int(input())
	# Número de bits a ser considerado na etapa de quantização.
	B = int(input())
	# Semente S para a função random.
	S = int(input())

	# Iniciando valor da semente para ramdom
	random.seed(S)

	# Gerando a imagem da cena.
	# Numpy array preenchidas com zeros de tamanho C x C.
	scene = np.zeros((C,C))
	# Gerando a cena de acordo com a função escolhida.
	functions = {
		1: f_one,
        2: f_two,
        3: f_three,
        4: f_four,
        5: f_five,
	}
	scene = apply(scene, functions.get(f), Q)

	print(scene[0,10])






if __name__ == "__main__":
    main()