# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251							 1/2018
# Trabalho 5		   Inpaiting using FFTS

import numpy as np
import imageio


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
	return np.sqrt( np.sum( np.power((H-Hr), 2) )/ (N*M) )

def normalize(m, min, max, dtype=None):
	"""Normaliza os valores de m entre min e max.

	Args:
		m (np.array): np array com valores a serem normalizados.
		min (float): minímo.
		max (float): máximo.

	Returns:
		np.array: np array com os valores normalizados.
	"""
	if(dtype == None):
		dtype = m.dtype
	return ( (max-min)*( (m-m.min() )/( m.max()-m.min() )) + min).astype(dtype=dtype)

def gerchberg_papoulis(g, m, T):
	"""Gerchberg Papoulis Algorithm

	Args:
	    g (np.array): image with noise
	    m (np.array): mask of noise
	    T (np.array): number of iterations

	Returns:
	    np.array:  image after algorithm.
	"""
	# g0 = g
	gk = g

	# Fourier transform to mask m
	M = np.fft.fft2(m)
	# Max value in M
	M_max = M.max()

	# Fourier transform to mean filter
	h = 7
	## Filling filter with zeros.
	me = np.zeros(g.shape)
	me[0:h, 0:h] = 1/(np.power(h, 2))
	## Filter DFT.
	ME = np.fft.fft2(me)

	# For each interation k in T
	for k in range(T):
		# Fourier transform to image gk
		G = np.fft.fft2(gk)
		# Filtering Gk
		G_max = G.max()
		## Removing in Gk frequencies with greater or equal than 90% and
		## frequencies with less or equal than 1% in Gk
		G[(G >= 0.9 * M_max) & (G <= 0.01 * G_max)] = 0

		# Convolution with mean filter.
		G = np.multiply(ME, G)

		# Getting inverse Fourier.
		gk = np.real(np.fft.ifft2(G))

		# Normalizing image gk.
		gk = normalize(gk, 0, 255, dtype=np.uint8)

		# Inserting pixels in k
		gk = np.multiply((1 - (m/255)), g ) + np.multiply((m/255), gk)

	return gk



def main():
	# Reading inputs.
	# Original image filename.
	imgo_fname = str(input().rstrip())
	# Deteriorated image filename.
	imgi_fname = str(input().rstrip())
	# Mask filename.
	imgm_fname = str(input().rstrip())
	# Interations number.
	T = int(input())

	# Reading images.
	imgo = imageio.imread(imgo_fname)
	imgi = imageio.imread(imgi_fname)
	imgm = imageio.imread(imgm_fname)

	# Executing the Gerchbeng Papoulis algorithm.
	imgout = gerchberg_papoulis(imgi, imgm, T)
	# Calculating error.
	error = RMSE(np.uint8(imgo), np.uint8(imgout))
	# Printing error with 5 decimal points
	print("%.5f" % error)

if __name__ == '__main__':
	main()