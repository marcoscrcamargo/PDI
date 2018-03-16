# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 							 1/2018
# Trabalho 1 			 Gerador de imagens

# imports
import numpy as np
import random
import math
import imageio

# Funções para geração da imagem da cena.

# Função 1 - x + y.
def f_one(m, Q):
	for (x, y), value in np.ndenumerate(m):
		m[x, y] = (x + y)
	return m

# Função 2 - |sin(x/Q) + sin(y/Q)|.
def f_two(m, Q):
	for (x, y), value in np.ndenumerate(m):
		m[x, y] = abs(math.sin(x/Q) + math.sin(y/Q))
	return m

# Função 3 - (x/Q) - sqrt(y/Q).
def f_three(m, Q):
	for (x, y), value in np.ndenumerate(m):
		m[x, y] = ((x/Q) - math.sqrt(y/Q))
	return m

# Função 4 - valor aleatório continuo entre 0 e 1.
def f_four(m, Q):
	for (x, y), value in np.ndenumerate(m):
		m[x, y] = random.random()
	return m


# Função 5 - caminho aleatório.
def f_five(m, Q):
	C = m.shape[0]
	moves = int(C*C/4)
	# Passo 1
	x = 0
	y = 0
	m[x, y] = 1.0

	for i in range(moves):
		dx  = random.randint(-1, 1)
		x = ((x + dx) % C)
		m[x, y] = 1.0

		dy = random.randint(-1, 1)
		y = ((y + dy) % C)
		m[x, y] = 1.0

	return m

# Função que calcula o máximo local dos d pontos a partir
# 	de (i,j) horizontal e verticalmente
def local_max(scene, i, j, d):
	return scene[int(i*d):int((i+1)*d):1, int(j*d):int((j+1)*d):1].max()

# Função que calcula os valores da imagem digital
#  a partir da cena utilizando o máximo local.
def gen_image(image, scene, B, f_max=local_max):
	d = scene.shape[0]/image.shape[0]

	# Normalizando valores entre 0 e 255
	# scene = (scene* ((np.iinfo(np.int8).max-np.iinfo(np.int8).min)/(scene.max() - scene.min())) )
	scene = 255*((scene)/(scene.max()))

	# Convertendo para int32
	scene = scene.astype('int')

	# Quantização
	print("gerando img")
	print(scene.min())
	print(scene.max())
	print(scene)
	scene = scene >> (8-B)

	# Máximo local
	for (i,j), value in np.ndenumerate(image):
		image[i,j] = (f_max(scene, i, j, d))

	return image

# Função que calcula o erro médio quadratico para duas matrizes.
def RMSE(g, R):
	return math.sqrt(sum(sum( (g-R)**2 )))

def main():
	# Recebendo parametros.
	# Nome do arquivo.
	fname = str(input().rstrip())
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

	# Iniciando valor da semente para random.
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
	scene =  functions.get(f)(scene, Q)
	imageio.imwrite("scene.png", scene)
	print("cena:")
	print(scene)
	print(scene.max())
	print(scene.min())
	# Gerando imagem digital a partir da cena.
	image = np.zeros((N, N))
	image = gen_image(image, scene, B)
	imageio.imwrite("image.png", image)

	# Comparação
	R = np.load(fname) # Carregando arquivo para comparação.

	error = RMSE(image, R)
	print("RESPOSTA")
	print(image)
	print("R-TRUE")
	print(R)
	print(error)

if __name__ == "__main__":
    main()