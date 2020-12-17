from math import sqrt
import csv
import os

os.chdir("/home/saul/")

# finding all prime numbers up to any given limit

def sieve(N):
	
	#print(N)
	#N = 100
	is_prime = (N + 1) * [True]
	#print('Is Prime   :', is_prime)
	#print(N, " ",int(sqrt(N)))
	for candidate in range(2, int(sqrt(N)) + 1):
		#print(N, " ",int(sqrt(N)))
		if is_prime[candidate]:
		   for witness in range(candidate * candidate, N + 1, candidate):
               is_prime[witness] = False
	return is_prime[N]


if __name__ == '__main__':
    primeNumbers = []
	#writer = csv.writer(open("sieve.csv", 'w'))
	for candidate in range(10, 100):
		if sieve(candidate):
	
			print(candidate)
			#writer.writerow([candidate])
            
