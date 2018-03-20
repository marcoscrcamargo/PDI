# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 							 1/2018
# Trabalho 1 			 Gerador de imagens

# imports
import numpy as np
import random
import math

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
		m[x, y] = abs((x/Q) - math.sqrt(y/Q))
	return m

# Função 4 - valor aleatório continuo entre 0 e 1.
def f_four(m, Q):
	for (x, y), value in np.ndenumerate(m):
		m[x, y] = random.random()
	return m


# Função 5 - caminho aleatório.
def f_five(m, Q):
	C = m.shape[0]
	moves = int(C*C/2)
	# Passo 1
	x = 0
	y = 0
	m[x, y] = 1.0

	for i in range(moves):
		# Passo em x
		dx  = random.randint(-1, 1)
		x = ((x + dx) % C)
		m[x, y] = 1.0
		# Passo em y
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
	# Calculo do d para a amostragem (C/N).
	d = int(scene.shape[0]/image.shape[0])

	# Normalizando valores entre 0 e 65535 (uint16 max).
	scene = np.iinfo(np.uint16).max*((scene-scene.min())/(scene.max()-scene.min()))

	# Amostragem - Máximo local
	for (i,j), value in np.ndenumerate(image):
		image[i,j] = f_max(scene, i, j, d)

	# Quantização
	image = 255*((image)/(image.max()))

	# Realizando shift para ficar somente com B bits.
	image = np.uint8(image) >> (8-B)

	return image

# Função que calcula o erro médio quadratico para duas matrizes.
def RMSE(g, R):
	return math.sqrt(sum(sum( (g-R)**2 )))

# Função principal do programa.
def main():
	# Recebendo parametros (inputs).
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
	# Numpy array para a cena preenchida com zeros de tamanho (C x C).
	scene = np.zeros((C,C))
	# Numpy array para a imagem preenchida com zeros de tamanho (N x N).
	image = np.zeros((N, N))

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

	# Gerando imagem digital a partir da cena.
	image = gen_image(image, scene, B)

	# Comparação
	R = np.load(fname) # Carregando arquivo para comparação.
	error = RMSE(image, R) # Calculo do erro
	print(error)


if __name__ == "__main__":
    main()