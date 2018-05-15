# !/usr/bin/python
# coding: utf-8

# Marcos Cesar Ribeiro de Camargo - 9278045
# SCC251							 1/2018
# Trabalho 5			  Image Restoration

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


def local_noise_reduction(Inoisy, sigma_noisy, n):
	h, w = Inoisy.shape
	d = int((n-1)/2)
	Iout = np.zeros((h, w))
	sigma_noisy_2 = np.power(sigma_noisy, 2)

	for i in range(h):
		for j in range(w):
			# Getting filter using wrap.
			N = Inoisy.take(range(i-d, i+d+1), mode='wrap', axis=0).take(
							range(j-d, j+d+1), mode='wrap', axis=1)
			# Calculating mean.
			mN = np.mean(N)
			# Calculation variance.
			sigmaN = np.var(N)
			# Calculationg output image.
			Iout[i, j] = Inoisy[i, j] - ((sigma_noisy_2/sigmaN) * (Inoisy[i, j] - mN))

	return Iout

def stage_A(Inoisy, n_cur, M, i, j):
	# Getting filter using clip.
	d = int((n_cur-1)/2)
	N = Inoisy.take(range(i-d, i+d+1), mode='clip', axis=0).take(
					range(j-d, j+d+1), mode='clip', axis=1)

	zmin = N.min()
	zmed = np.median(N)
	zmax = N.max()

	A1 = (zmed - zmin)
	A2 = (zmed - zmax)
	if(A1 > 0 and A2 < 0):
		# Stage B
		B1 = Inoisy[i, j] - zmin
		B2 = zmed - zmax
		if (B1 > 0 and B2 < 0):
			return Inoisy[i, j]
		else:
			return zmed
	else:
		n_cur = n_cur + 1
		if(n_cur <= M):
			return stage_A(Inoisy, n_cur, M, i, j)
		else:
			return zmed

def median_adaptative(Inoisy, M, n):
	h, w = Inoisy.shape
	Iout = np.zeros((h, w))

	for i in range(h):
		for j in range(w):
			# Stage A
			Iout[i, j] = stage_A(Inoisy, n, M, i, j)

	return Iout


def counter_harmonic_mean(Inoisy, Q, n):
	h, w = Inoisy.shape
	# Calculating bounds
	d = int((n-1)/2)
	Iout = np.zeros((h, w))

	# Zero padding
	Inoisy_pad = np.pad(Inoisy, d, mode='constant', constant_values=(0))

	for i in range(h):
		for j in range(w):
			# Getting filter using zero padding.
			N = Inoisy_pad[i:i+2*d+1, j:j+2*d+1]
			# Calculationg output image.
			Iout[i, j] = np.sum(np.power(N, Q+1))/np.sum(np.power(N, Q))

	return Iout



def main():
	## Reading inputs.
	# Original image file name.
	Icomp_fname = str(input().rstrip())
	# Noisy image file name.
	Inoisy_fname = str(input().rstrip())
	# Method.
	method = int(input().rstrip())
	# Filter size.
	n = int(input().rstrip())
	# Specific parameters to each method.
	P = float(input().rstrip())

	# Reading images.
	Icomp = imageio.imread(Icomp_fname)
	Inoisy = imageio.imread(Inoisy_fname)

	# Appliying the chosen filter.
	filter = {
		1: local_noise_reduction,
		2: median_adaptative,
		3: counter_harmonic_mean,
	}
	Iout = filter.get(method)(Inoisy, P, n)

	# Calculating error.
	error = RMSE(np.uint8(Icomp), np.uint8(Iout))

	# Printing error with 5 decimal points
	print("%.5f" % error)

if __name__ == '__main__':
	main()