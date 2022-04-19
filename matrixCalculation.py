# matrix calculation
from numpy import matrix # Returns a matrix from an array-like object, or from a string of data
from numpy import linalg # The Linear Algebra module of NumPy offers various methods to apply linear algebra on any numpy array

def createMatrix():
	A = matrix([[1,2,3], [11,12,13], [21, 22,23]])
	x = matrix( [[1],[2],[3]] )
	y = matrix( [[1,2,3]] )

	print('Matrix of A :',A)
	print(A*x)
	calculateMatrix(A, x, y)

def calculateMatrix(A,x,y):
	multip = A*x
	print(A + y)
	print(A*x)

	print('Multiplication :', multip)
	print('Transpose of A : ', A.T)
	print('Inverse of A :', A.I)
	print ('Solve Linear Equation of A an X :', linalg.solve(A, x))
	print(A[2,1])
	
if __name__ == '__main__':
	createMatrix()
