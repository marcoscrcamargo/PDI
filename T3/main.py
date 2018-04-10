# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 							 1/2018
# Trabalho 3				   Filtragem 1D

import numpy as np
import imageio
import math


def G1D(x, sigma):
	"""Função que retorna o valor da função gaussiana para um dado x e sigma.

	Args:
	    x (float): valor x.
	    sigma (float): valor sigma.

	Returns:
	    float: resultado da função gaussiana.
	"""
	return (math.exp( (-1/2) * ((x**2)/(sigma**2)) )/( math.sqrt(2*math.pi)*sigma ) )


def get_filter(filt, n):
	'''
	Função que retorna um np.array com pesos lidos do teclado caso o filtro seja arbitrario (filt=1) ou
	uma distribuição gaussiana (filt=2).

	Args:
	    filt (int): tipo de filtragem
	    n (int): tamanho do filtro

	Returns:
	    np.array: vetor com a mascara para filtragem ser realizada.
	'''
	if(filt == 1):
		# Lendo array como string.
		str = input().rstrip().split(' ')
		# Convertendo para um array
		f = [float(i) for i in str]
		# np.array
		f = np.array(f)

	elif(filt == 2):
		# Lendo sigma.
		sigma = float(input())
		# Calculo dos extremos do vetor.
		d = int((n-1)/2)
		# Lista de -d até d
		f = np.arange(-d, d+1)
		# Transformando G1D em uma função vetorial.
		v_G1D = np.vectorize(G1D)
		# Aplicando a gaussiana no array.
		f = v_G1D(f, sigma=sigma)

	return f

def RMSE(H, Hr):
	'''
	Função que calcula o erro médio quadratico para duas imagens.
	Args:
		H (np.array): imagem 1
		Hr (np.array): imagem 2
 	Returns:
		float: Erro médio quadratico para duas imagens.
	'''
	return math.sqrt( ((sum( (H-Hr)**2 )))/(Hr.shape[0]**2) )

def cross_correlation_point(f, mask, x):
	"""Calcula a Cross Correlation em um vetor 1D utilizando imagem wrap
	para tratar as bordas para um ponto.
	Args:
	    f (np.array): vetor que sera utilizado.
	    mask (np.array): mascara a ser aplicada no vetor.
	    x (int): posição central em que a mascara será posicionada.
	Returns:
	    float: valor resultante da correlação no ponto.
	"""
	n = mask.shape[0]
	N = f.shape[0]
	d = int((n-1)/2)
	# Pegando submatriz de x-d até x+d, considerando um vetor circular.
	idx = range(x-d, x+d+1)
	sub_f = f.take(idx, mode="wrap")
	return int( np.sum(np.multiply(sub_f, mask)) )

def cross_correlation(f, mask):
	"""Calcula a Cross Correlation em um vetor 1D utilizando imagem wrap
	para tratar as bordas.
	Args:
	    f (np.array): vetor que sera utilizado.
	    mask (np.array): mascara a ser aplicada no vetor.
	Returns:
	    np.array: array resultante da correlação.
	"""
	N = f.shape[0]
	g = np.empty(N, dtype=np.int)
	for i in range(N):
		g[i] = cross_correlation_point(f, mask, i)
	return g

def main():
	## Recebendo parâmetros.
	# Nome da imagem para ser filtrada.
	img_name = str(input().rstrip())
	# Opção de escolha do filtro.
	opt = int(input())
	# Tamanho do filtro.
	n = int(input())
	# Parâmetros especificos para cada tipo de filtro.
	weights = get_filter(opt, n)
	# Dominío da filtragem.
	D = int(input())

	# Lendo imagem.
	I = imageio.imread(img_name)

	# Transformando I para vetor 1D
	I = I.reshape(I.shape[0]*I.shape[0])
	# Aplicando a transformação.
	I_hat = cross_correlation(I, weights)

	# Imprimindo o erro.
	error = RMSE(I, I_hat)
	print(error)

if __name__ == '__main__':
	main()