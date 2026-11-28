import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	n = len(A)
	for i in range(n):
		if A[i][i] == 0:
			for j in range(n):
				if A[j][i] != 0:
					a = A[i].copy()
					A[i] = A[j]
					A[j] = a

					a = b[i]
					b[i] = b[j]
					b[j] = a
					break
		for j in range(i):
			if A[j][j] == 0:
				continue
			radio = A[i][j] / A[j][j]

			for k in range(n):
				A[i][k] -= radio * A[j][k]
			b[i] -= radio * b[j]
#	print(b)
#	print(A)
	for idx in range(n):
		i = n - idx - 1
		b[i] = b[i] / A[i][i]
		# print(b[i])

		for j in range(i):
			b[j] -= A[j][i] * b[i];
	# print(b)
	# print(A)
	return b;


