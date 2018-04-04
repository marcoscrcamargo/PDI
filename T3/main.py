# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 							 1/2018
# Trabalho 3				   Filtragem 1D

import numpy as np
import imageio

def get_filter(filt, n):
	'''
	Função que retorna um np.array com pesos lidos do teclado caso o filtro seja arbitrario (filt=1) ou
	uma distribuição gaussiana (filt=2).
	filt = tipo de filtragem
	n 	 = tamanho do filtro
	'''
	if(filt == 1):
		str = input().rstrip().split(' ')
		f = [float(i) for i in str]
	elif(filt = 2):
		sigma = float(input())


	return f

def RMSE(H, Hr):
 	'''
 	Função que calcula o erro médio quadratico para duas imagens.
	H = np.array contendo a imagem
	Hr = np.array contendo a imagem
	'''
	return math.sqrt( (sum(sum( (H-Hr)**2 )))/(Hr.shape[0]**2) )

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


if __name__ == '__main__':
	main()