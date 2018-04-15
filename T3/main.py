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
	N, M = H.shape
	return math.sqrt( sum(sum( (H-Hr)**2 ))/(N*M) )

def conv_point(f, mask, x):
	"""Calcula a Convulução em um vetor 1D utilizando imagem wrap
	para tratar as bordas para um ponto.
	Args:
		f (np.array): vetor que sera utilizado.
		mask (np.array): mascara a ser aplicada no vetor já invertida.
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

def image_convolution(f, mask):
	"""Calcula a Convulução em um vetor 1D utilizando imagem wrap
	para tratar as bordas.
	Args:
		f (np.array): vetor que sera utilizado.
		mask (np.array): mascara a ser aplicada no vetor.
	Returns:
		np.array: array resultante da correlação.
	"""
	N = f.shape[0]
	g = np.empty(N, dtype=np.int)

	# Invertendo a mascara para realizar a convolução.
	mask = np.flip(mask, 0)

	for i in range(N):
		g[i] = conv_point(f, mask, i)
	return g

def normalize(m, min, max):
	"""Normaliza os valores de m entre min e max.

	Args:
		m (np.array): np array com valores a serem normalizados.
		min (float): minímo.
		max (float): máximo.

	Returns:
		np.array: np array com os valores normalizados.
	"""
	return ( (max-min)*( (m-m.min() )/( m.max()-m.min() )) + min)


def spatial_domain(I, weights):
	"""Realiza a convulução no domínio espacial para a imagem I com a mascara weights.

	Args:
		I (np.array): Imagem a qual sera realizada a convolução
		weights (np.array): mascara 1D que será utilizada na convolução.

	Returns:
		np.array: Imagem 2D com as mesmas dimenções de I, já convolucionada e normalizada
	"""
	# Salvando as informações sobre o formato do shape da image.
	N, M = I.shape

	# Transformando I para vetor 1D
	I = I.flatten()

	# Aplicando a transformação 1D.
	I_hat = image_convolution(I, weights)

	# Normalizando a saída entre 0 e 255
	I_hat = normalize(I_hat, 0, 255)

	# Convertendo a saída para 2D novamente.
	I_hat = I_hat.reshape((N, M))

	return I_hat



def DFT1D(A):
	"""Função que calcula a Transformada Discreta de Fourier para um vetor 1D.
	Código baseado na aula, disponível em:
	https://github.com/maponti/dip_code_2018/blob/master/04b_fourier_transform_1d.ipynb
	Args:
	    A (np.array): Vetor de entrada

	Returns:
	    np.array: Transformada Discreta de Fourier
	"""
	F = np.zeros(A.shape, dtype=np.complex64)
	n = A.shape[0]
	# Índices de x entre 0 e n-1.
	x = np.arange(n)
	# Para cada frequencia u, computamos de forma vetorial para x e somamos em F[u]
	for u in np.arange(n):
		F[u] = np.sum(A*np.exp( (-1j * 2 * np.pi * u*x) / n ))

	return F

def IDFT1D(F):
	"""Função que calcula a inversa da Transformada Discreta de Fourier utilizando utilizando a função
	DFT1D, o método descrito em:
	https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Expressing_the_inverse_DFT_in_terms_of_the_DFT

	Args:
	    F (np.array): Transformada Discreta de Fourier
	Returns:
	    np.array: Inversa da Transformada Discreta de Fourier
	"""
	return np.real(DFT1D(np.concatenate( ([F[0]], F[-1:0:-1] ), axis=0))/F.shape[0])

def frequency_domain(I, weights):
	"""Realiza a convulução no domínio espacial para a imagem I com a mascara weights.

	Args:
		I (np.array): Imagem a qual sera realizada a convolução
		weights (np.array): mascara 1D que será utilizada na convolução.

	Returns:
		np.array: Imagem 2D com as mesmas dimenções de I, já convolucionada e normalizada
	"""
	# Salvando as informações sobre o formato do shape da image.
	N, M = I.shape

	# Transformando I para vetor 1D.
	I = I.flatten()

	# Preenchendo weights com zeros.
	weights = np.concatenate((weights, np.zeros((N*M)-weights.shape[0])), axis=0)

	# Calculo das transformadas de Fourier para F e W.
	F = DFT1D(I)
	W = DFT1D(weights)

	# Calculo da Transformada de Fourier Inversa para o produto ponto-a-ponto de F e W.
	I_hat = IDFT1D(W*F)

	# Normalizando a saída entre 0 e 255
	I_hat = normalize(I_hat, 0, 255)

	# Convertendo a saída para 2D novamente.
	I_hat = I_hat.reshape((N, M))

	return I_hat

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

	domain = {
		1: spatial_domain,
		2: frequency_domain,
	}
	# Executando a convolução conforme o domínio escolhido.
	I_hat = domain.get(D)(I, weights)

	# Imprimindo o erro.
	error = RMSE(np.uint8(I), np.uint8(I_hat))
	print(error)

if __name__ == '__main__':
	main()