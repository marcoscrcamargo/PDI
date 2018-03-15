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
def gen_scene(m, f, Q):
	for (x,y), value in np.ndenumerate(m):
		m[x,y] = f(x, y, Q)

	return m

# Função que calcula o máximo local dos d pontos a partir
# 	de (i,j) horizontal e verticalmente
def local_max(scene, i, j, d):
	return scene[i*d:(i+1)*d:1, j*d:(j+1)*d:1].max()

# Função que calcula os valores da imagem digital
#  a partir da cena utilizando o máximo local.
def gen_image(image, scene, B, f_max=local_max):
	d = scene.shape[0]/image.shape[0]

	for (i,j), value in np.ndenumerate(image):
		image[i,j] = np.uint8(f_max(scene, i, j, d)) 
						& (255 >> 8-B) # Mudar esse 8 hardcoded ?

	return image

def RMSE(g, R):
	return math.sqrt(sum(sum( (g-R)**2 )))

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

	# Definindo a possibilidade das funções para gerar a cena.
	functions = {
		1: f_one,
        2: f_two,
        3: f_three,
        4: f_four,
        5: f_five,
	}
	# Gerando a cena de acordo com a função escolhida.
	scene = gen_scene(scene, functions.get(f), Q)

	# Gerando imagem digital a partir da cena.
	image = np.zeros((N, N), dtype=np.uint8)
	image = gen_image(image, scene, B)

	# Comparação

if __name__ == "__main__":
    main()