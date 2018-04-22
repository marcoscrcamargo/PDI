# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251 							 1/2018
# Trabalho 4				   Filtragem 2D

import numpy as np
import imageio

def LoG2D(x, y, sigma):
	"""Função que calcula o valor da Laplaciana da Gaussiana para um dado x, y e sigma

	Args:
	    x (float): valor x de entrada.
	    y (float): valor y de entrada.
	    sigma (float): sigma da função

	Returns:
	    float: resultado da LoG
	"""
	delta = (x**2 + y**2)/(2*sigma**2)
	return ((-1)/(np.pi*sigma**4)) * (1 - delta) * np.exp(-delta)

def get_weights(method):
	"""Função que recebe da os parametros para o calculo dos pesos para convolução

	Args:
	    method (int): método para geração dos parâmetros.

	Returns:
	    np.array: array contendo os pesos.
	"""
	# Recebendo 2 parâmetros
	weights = 0
	if(method == 1):
		string = input().rstrip().split()
		h = int(string[0])
		w = int(string[1])
		weights = np.zeros((h, w))
		# Lendo valores da entrada.
		for i in range(h):
			# Lendo linha em forma de string.
			string = input().rstrip().split()
			for j in range(w):
				# Convertendo para matriz.
				weights[i, j] = float(string[j])

	elif(method == 2):
		# Parametros do método.
		n = int(input())
		sigma = float(input())
		# Obtendo o espaço linear entre -5 e 5 com n divisões.
		s = np.linspace(-5.0, 5.0, n)
		# Calculando os valores para LoG
		weights = np.zeros((n, n))
		for i in range(n):
			for j in range(n):
				weights[i, j] = LoG2D(s[i], s[j], sigma)
		# Normalizando.
		pos = sum(weights[weights>0])
		neg = sum(weights[weights<0])
		f = weights*(-pos/neg)
		# Se negativo normaliza.
		weights[weights<0] = f[weights<0]

	return weights

def get_cuts():
	"""Retorna os bounds para os cortes.

	Returns:
	    np.array: array com 4 limites para os cortes (Hlb, Hub, Wlb, Wub)
	"""
	# Recebendo string com valores.
	string = input().rstrip().split()
	# Transformando em array e retornando.
	bounds = [float(i) for i in string]
	return bounds

def cut(I, Hlb, Hub, Wlb, Wub):
	"""Realiza o corte proporcionalmente com os limitantes.
	Args:
	    I (np.array): Imagem original
	    Hlb (float): limitante inferior da linha
	    Hub (float): limitante superior da linha
	    Wlb (float): limitante inferior da coluna
	    Wub (float): limitante superior da coluna
	Returns:
	    np.array: Imagem cortada.
	"""
	H, W = I.shape
	# Realizando corte 1 pela simetria.
	I_cut = I[0:int(H/2), 0:int(W/2)]
	# Realizando corte 2 pelos limitantes.
	H, W = I_cut.shape
	return I_cut[int(Hlb*H):int(Hub*H), int(Wlb*W):int(Wub*W)]

def freq_conv(I, weights):
	"""Realiza a convolução parcial no domínio das frequências
	Args:
	    I (np.array): Imagem de entrada.
	    weights (np.array): pesos da convolução.
	Returns:
	    np.array: array de valores complexos com a DFT da convolução da imagem I.
	"""
	## Preenchendo weights com zeros.
	# Shape de weights
	h, w = weights.shape
	# W com tamanho da imagem
	W = np.zeros(I.shape)
	# Colocar weights em W
	W[0:h, 0:w] = weights

	# Calculo das transformadas de Fourier para F e W.
	F = np.fft.fft2(I)
	W = np.fft.fft2(W)

	# Calculo do produto ponto-a-ponto de F e W.
	I_out = np.multiply(W, F)

	return I_out


def convolution2D(f, w):
	"""Realiza a convolução 2D para uma imagem f e filtro w

	Args:
	    f (np.array): imagem de entrada
	    w (np.array): filtro da convolução

	Returns:
	    np.array: Resultado da convolução.
	"""
	# Obtendo shapes da imagem e do filtro.
	N, M = f.shape
	n, m = w.shape
	# Calculo do centro do filtro.
	a = int((n-1)/2)
	b = int((m-1)/2)
	# Invertendo o filtro para convolução.
	w_flip = np.flip(np.flip(w, 0), 1)
	# Criando matriz de saída.
	g = np.zeros((N, M))
	f = np.pad(f, (a, b), mode='constant', constant_values=(0))
	# Para cada pixel da imagem, realiza a convolução.
	for i in range(1, N+1):
		for j in range(1, M+1):
			# Submatriz para multiplicação.
			sub_f = f[i-a:i+a+1, j-b:j+b+1]
			# Calculo da convolução no ponto.
			g[i-1, j-1] = np.sum(np.multiply(sub_f, w_flip))
	return g

def sobel_opr(I):
	"""Realiza o operador sobel na imagem I com dois filtros arbitrários Fx e Fy.

	Args:
	    I (np.array): imagem de entrada para o operador.

	Returns:
	    np.array: DFT do resultado do operador sobel.
	"""
	# Filtros arbitrários.
	Fx = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape((3,3))
	Fy = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3,3))
	# Calculo da convolução com filtro Fx.
	Ix = convolution2D(I, Fx)
	# Calculo da convolução com filtro Fy.
	Iy = convolution2D(I, Fy)
	# Calculo da imagem final.
	I_out = np.sqrt(np.power(Ix, 2) + np.power(Iy, 2))
	# Retorno da DFT.
	return np.fft.fft2(I_out)

def KNN(X, Y, query, k=1, EF=np.linalg.norm):
	"""Algoritmo K-Nearest Neighbors

	Args:
	    X (np.array): dataset.
	    Y (np.array): labels do dataset.
	    query (TYPE): ponto a ser classificado.
	    k (int, optional): número de vizinhos mais próximos (padrão 1)
	    EF (func, optional): função para o calculo da distância (padrão distância euclidiana).

	Returns:
	    (int, np.array): label encontrado, ids dos k mais próximos.
	"""
	# Calculo das distancias de X para a query
	dist = [EF(xi-query) for xi in X]
	# IDs dos K mais próximos
	ids = np.argsort(dist)[0:k]
	# Classe mais frequente nos k mais próximos
	label = np.bincount(Y[ids]).argmax()

	return (label, ids)

def main():
	## Recebendo parâmetros.
	# Nome da imagem para ser filtrada.
	I_name = str(input().rstrip())
	# Opção de escolha do filtro.
	method = int(input())
	# Parametros para o filtro.
	weights = get_weights(method)
	# Números reais para os cortes
	Hlb, Hub, Wlb, Wub = get_cuts()
	# Nome do arquivo .npy com o dataset
	X_name = str(input().rstrip())
	# Nome do arquivo .npy com os labels do dataset
	Y_name = str(input().rstrip())

	# Lendo imagem.
	I = imageio.imread(I_name)
	# Aplicando filtro na imagem no domínio das frequencias
	if(method == 3):
		I_out = sobel_opr(I)
	else:
		I_out = freq_conv(I, weights)

	# Realizando Cortes
	I_cut = cut(I_out, Hlb, Hub, Wlb, Wub)
	## Classificação
	# Lendo dataset e labels
	X = np.load(X_name)
	Y = np.load(Y_name).astype(dtype="int32")
	# Convertendo I para 1D
	I_flat = I_cut.flatten()
	# Classificando com KNN (k=1)
	Y_hat, pos = KNN(X=X, Y=Y, query=I_flat, k=1)
	# Imprimindo rótulo encontrato
	print(Y_hat)
	print(pos[0])

if __name__ == '__main__':
	main()