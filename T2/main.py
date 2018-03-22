# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 							 1/2018
# Trabalho 2 			 	super-resolução

import numpy as np
import imageio
import math

# Função para a leitura das 4 imagens de baixa qualidade.
def read_images(fname):
	L = []
	for i in range(0,4):
		# Lendo as imagens, e colocando em um array.
		L.append( imageio.imread(fname + str(i+1) + ".png") )

	# Retornando as imagens lidas no array.
	return L #(L[0], L[1], L[2], L[3])

# Função para o calculo do histograma acumulado.
def ha(img):
	# Calculando os ids, e a quantidade de vezes que cada valor se repete para o id.
	(ids, qnt) = np.unique(img, return_counts=True)
	# Inicia uma matriz com 255 valores em zero.
	h = np.zeros(255)

	# Calculo do histograma.
	for i in range(ids.shape[0]):
		h[int(ids[i])] = qnt[i] 

	# Calculo do histograma acumulado.
	ha = np.cumsum(h)

	return ha

# Função que gera uma imagem H com super-resolução a partir de um
# 	array de imagens L.
def SR(L):
	# Tamanho das imagens L.
	N = L[0].shape[0]
	# Gerando a imagem H
	H = np.zeros((2*N, 2*N))

	# Percorrendo quadrantes.
	for (di, dj) in np.ndindex((2,2)):
		# Percorrendo cada imagem.
		for (i, j), v in np.ndenumerate(L[dj*2+di]):
			# Atribuindo super-resolução.
			H[2*i+dj, 2*j+di]= L[dj*2+di][i, j]

	return H

# Função que calcula o erro médio quadratico para duas imagens.
def RMSE(H, Hr):
	return math.sqrt( (sum(sum( (H-Hr)**2 )))/(Hr.shape[0]**2) )


def main():
	# Recebendo parametros (inputs).
	## Nome do arquivo para imagem de baixa resolução.
	imglow = str(input())
	## Nome do arquivo para imagem de alta resolução.
	imghigh = str(input())
	## Método de realce.
	opt = int(input())
	## Parâmetro do realce gamma.
	gamma = int(input())

	# Lendo as imagens de baixa qualidade.
	L = read_images(imglow)

	# Gerando imagem com super-resolução.
	H = SR(L)	

	# Abrindo imagem de alta qualidade.
	Hr = imageio.imread(imghigh + ".png")
	# Comparando imagens, e imprimindo o valor com 4 casas decimais.
	print("%.4f" % RMSE(H, Hr))

if __name__ == '__main__':
	main()