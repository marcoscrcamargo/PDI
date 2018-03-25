# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 							 1/2018
# Trabalho 2 			 	Super-resolução

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

	# Inicia uma matriz com 256 valores em zero.
	h = np.zeros(256)

	# Calculo do histograma.
	for i in range(ids.shape[0]):
		h[int(ids[i])] = qnt[i]

	# Calculo do histograma acumulado.
	ha = np.cumsum(h)

	return ha

# Função que faz a equalização por histograma dado uma imagem e histograma.
def histogram_equalization(img, hist):
	L = 256
	N = img.shape[0]

	for (i, j), v in np.ndenumerate(img):
		img[i, j] = ((L - 1)/(N**2))*hist[img[i, j]]

	return img

# Função que faz o ajuste gamma sobre a imagem img.
def gamma_adjust(img, gamma):
	return np.uint8(255.0*((img/255.0)**gamma))

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
		for (i, j), v in np.ndenumerate(L[di*2+dj]):
			# Atribuindo super-resolução.
			H[2*i+di, 2*j+dj]= L[di*2+dj][i, j]

	return H

# Função que calcula o erro médio quadratico para duas imagens.
def RMSE(H, Hr):
	return math.sqrt( (sum(sum( (H-Hr)**2 )))/(Hr.shape[0]**2) )

def main():
	# Recebendo parametros (inputs).
	## Nome do arquivo para imagem de baixa resolução.
	imglow = str(input().rstrip())
	## Nome do arquivo para imagem de alta resolução.
	imghigh = str(input().rstrip())
	## Método de realce.
	opt = int(input())
	## Parâmetro do realce gamma.
	gamma = float(input())

	# Lendo as imagens de baixa qualidade.
	L = read_images(imglow)
	# Método de realce 1.
	# Função de transferencia individual.
	if(opt == 1):
		for i in range(len(L)):
			hist = ha(L[i])
			L[i] = histogram_equalization(L[i], hist)

	# Método de realce 2.
	# Função de transferencia conjunta.
	elif(opt == 2):
		hist = 0
		for Li in L:
			hist += ha(Li)

		for i in range(len(L)):
			L[i] = histogram_equalization(L[i], hist)

	# Método de realce 3.
	# Função de ajuste gamma.
	elif(opt == 3):
		for i in range(len(L)):
			L[i] = gamma_adjust(L[i], gamma)

	# Gerando imagem com super-resolução.
	H = SR(L)

	# Abrindo imagem de alta qualidade.
	Hr = imageio.imread(imghigh + ".png")
	# Comparando imagens, e imprimindo o valor com 4 casas decimais.
	print("%.4f" % RMSE(H, Hr))

if __name__ == '__main__':
	main()