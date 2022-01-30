from math import sqrt, inf
import string
from itertools import permutations
import clipboard
import webbrowser
from functools import reduce
import operator
from collections import Counter
import requests

def digits(base=10):
	return [str(i) for i in range(base)]
	
	
def factors(n):
	facts = set([1,n])
	for i in range(2, int(n**0.5)+1):
		if n%i == 0:
			facts.update((i, n//i))
	return facts
	
def primes(maxval = 1000000):
	prime = [False,False]+[False,True]*(maxval//2)
	prime[2] = True
	prime = prime[:maxval+1]
	for i in range(3,maxval,2):
		if prime[i]:
			for j in range(i*2,maxval,i):
				prime[j] = False
	i = 1
	for i,bool in enumerate(prime):
		if bool:
			yield i
			
def square(maxval = 1000000):
	for i in range(1,int(sqrt(maxval))):
		yield i**2
		
def prod(iterable):
	return reduce(operator.mul, iterable, 1)
	
def factorial(n):
	return prod(range(1,n+1))
	
def quadratic(a,b,c):
	d = sqrt(b**2-4*a*c)
	return ((-b+d)/(2*a), (-b-d)/(2*a))
	
def isprime(n):
	if n == 1:
		return False
	elif n<3:
		return True
	for i in range(2,int(sqrt(n+1))+1):
		if n%i == 0:
			return False
	return True
	
def isbouncy(n):
	s = str(n)
	
def ispandigital(n, min = 1, max=9):
	digits = string.digits[min:max+1]
	return len(str(n)) == max -min+1 and all([i in str(n) for i in digits])
	
def pandigitals(min=1,max=9):
	for i in permutations(string.digits[min:max+1]):
		yield int(''.join([d for d in i]))
		
def euclidtriangles(maxperim = 1000):
	maxi = int(sqrt(maxperim/2))
	for m in range(2,maxi):
		for n in range(1,m):
			if (m^n)&0b1:
				a = m ** 2 - n ** 2
				b = 2*m*n
				c = m ** 2 + n ** 2
#                               print(m,n)
				yield (a, b, c)
				
				
def hexagonals():
	i=0
	while True:
		i+=1
		yield int(i*(2*i-1))
		
def pentagonals():
	i=0
	while True:
		i+=1
		yield int(i*(3*i-1)/2)
		
def triangulars():
	i=0
	while True:
		i+=1
		yield int(i*(i+1)/2)
		
def ishex(n):
	a = 2
	b = -1
	c = -n
	return any([i % 1 == 0 for i in quadratic(a,b,c) if i>0])
	
def ispent(n):
	a = 3/2
	b = -1/2
	c = -n
	return any([i % 1 == 0 for i in quadratic(a,b,c) if i>0])
	
def istriangle(n):
	a = 1/2
	b = 1/2
	c = -n
	return any([i % 1 == 0 for i in quadratic(a,b,c) if i>0])
	
def rotations(n: int):
	n = str(n)
	return [int(n[i:]+n[:i]) for i in range(len(n))]
	
def truncated(n):
	s = str(n)
	return [int(s[:i]) for i in range(1,len(s))]+[int(s[i:]) for i in range(1,len(s))]
	
def prob11():
	grid = '''08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
	49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
	81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
	52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
	22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
	24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
	32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
	67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
	24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
	21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
	78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
	16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
	86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
	19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
	04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
	88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
	04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
	20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
	20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
	01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48'''.splitlines()
	m = [s.split(' ') for s in grid]
	rows = []
	cols = []
	diagdr = []
	diagdl = []
	for r in range(len(m)):
		for c in range(len(m[0])):
			try:
				rows.append(prod([int(m[r][c + i]) for i in range(4)]))
			except IndexError:
				pass
				
			try:
				cols.append(prod([int(m[r+i][c]) for i in range(4)]))
			except IndexError:
				pass
				
			try:
				diagdr.append(prod([int(m[r+i][c+i]) for i in range(4)]))
			except IndexError:
				pass
				
			try:
				diagdl.append(prod([int(m[r+i][c-i]) for i in range(4)]))
			except IndexError:
				pass
				
	print('rows \n',max(rows))
	print('cols \n',max(cols))
	print('diagdr \n',max(diagdr))
	print('diagdl \n',max(diagdl))
	
def prob15():
	n = 20
	n+=1
	m = [[0 for i in range(n+1)] for j in range(n+1)]
	m[0][0]=1
	print(n-1)
	for i in range(1,n+1):
		for r in range(i):
			m[r][i]=m[r-1][i]+m[r][i-1]
		for c in range(i+1):
			m[i][c]=m[i][c-1]+m[i-1][c]
	return (m[n-1][n-1])
#       print(m)
	def walk(i,j):
		nonlocal fin
		if i<n:
			fin= walk(i+1,j)
		if j<n:
			fin = walk(i,j+1)
		if i == j == n:
			fin += 1
		return fin
	for n in range(1,2):
		fin = 0
		fin = walk(0,0)
		#print(n,fin,n*2**n)
	#return fin
	
def prob17():
	def spell(n):
		n = str(n)
		digis = ['','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
		tens = {1:'',2:'twenty ',3:'thirty ',4:'forty ',5:'fifty ',6:'sixty ',7:'seventy ',8:'eighty ',9:'ninety '}
		s = ''
		isand = False
		try:
			if n[-4]:
				s += digis[int(n[-4])]+' thousand '
				n = str(int(n[-3:]))
				isand = True
		except IndexError:
			pass
		try:
			if n[-3]:
				s += digis[int(n[-3])]+' hundred '
				n = str(int(n[-2:]))
				isand = True
		except IndexError:
			pass
		if int(n):
			if isand:
				s +='and '
		try:
			if n[-2]:
				s += tens[int(n[-2])]
				if n[-2] == '1':
					pass
				else:
					n = str(int(n[-1:]))
		except IndexError:
			pass
		if int(n)<20:
			s += digis[int(n)]
		return s
	c = ''
	for i in range(1,1001):
		c +=(str(spell(i)).replace(' ',''))
	return len(c)

def prob23():
	maxi = 28123
	r = iter(range(1,maxi))
	def solve(r):
		abundant = []
		summ = 0	
		for i in r:
	#		if i == sum(factors(i)) -i:
	#			print(i, 'perfect')
			if i < sum(factors(i)) -i:
				abundant.append(i)
			for a in abundant:
				if i-a in abundant:
					try:
						next(r)
					except StopIteration:
						return abundant, summ			
			summ += i
	abundant, summ = solve(r)
	print(len(abundant), 'abundant numbers')
	return summ
	
def prob31():
	coins = [1,2, 5,10,20,50,100]
	wins = set()
	path = []
	def pickcoin(p,cs):
		nonlocal wins
		val = cs[0]
		for i in range((200-sum(p))//val):
			path = p.copy()
			path.extend([val for j in range(i)])
			if sum(path)< 200:
				if len(cs)>1:
					path = pickcoin(path, cs[1:])
			if sum(path) == 200:
#                               print(path)
				wins.add(tuple(sorted(path)))
			if sum(path) > 200:
				#print('too much',path)
				pass
		if 'path' in locals():
			return path
		else:
			return p
			
#                               return s,w
#                       if s > 200:
#                               return s,w

	path = pickcoin(path,coins)
	print(len(wins)+1) #2€ coin
	return len(wins)+1
	
def prob32():
	sums = set()
	for m1 in range(2,100):
		for m2 in range(2,10000):
			p= m1   *m2
			s = ''.join([str(x) for x in [m1,m2,p]])
			if ispandigital(s):
				print(m1,m2,p)
				sums.add(p)
	print(f'sum = {sum([i for i in sums])}')
	
	
def prob35():
	def rotations(n:str):
		return [n[i:]+n[:i] for i in range(len(n))]
	summ = 1 # 2
	maxi = 100000
	prime = [False,False]+[False,True]*(maxi//2)
	for i in range(3,maxi,2):
		if prime[i]:
			for j in range(i*2,maxi,i):
				prime[j] = False
				
	print(sum(prime))
	
	for i in range(3,maxi,2):
		if prime[i]:
			if all([prime[j] for j in [int(''.join(k)) for k in rotations(str(i))]]):
				summ+=1
				print([int(''.join(k)) for k in rotations(str(i))])
				print(i)
				
	return summ, prime
	
def prob37():
	ts=[]
	summ = 0
	maxi = 1000000
	prime = [True]+[True,False]*(maxi//2)
	for i in range(3,maxi,2):
		if prime[i]:
			if all([isprime(x) for x in truncated(i)]):
				ts.append(i)
				summ += i
				print(ts)
			for j in range(i*2,maxi,i):
				prime[j] = False
	return summ
	
	
def prob39():
	pmax = 1000
	triangles = {}
	for tup1 in euclidtriangles():
		p = sum(tup1)
		tups = [tuple(int(i*s) for s in tup1) for i in range(1,(pmax+p)//p)]
		for tup in tups:
			p = sum(tup)
			if p in triangles.keys():
				if tup not in triangles[p]:
					triangles[p].append(tup)
					
			else:
				triangles[p] = [tup]
			print(tup, p)
	maxi = 0
	maxp = None
	for key, item in triangles.items():
		if len(item) > maxi:
			maxi = len(item)
			maxp = key
			print(maxi, maxp)
	return maxp
	
	
def prob41():
	for i in pandigitals(1,7):
		if isprime(i):
			print(i)
			
def prob42():
	r = requests.get(('https://projecteuler.net/project/resources/p042_words.txt'))
	l=r.text.replace('"','').split(',')
	values= [sum([ord(s)-64 for s in w]) for w in l]
	return len([v for v in values if istriangle(v)])
	
def prob43():
	def divisibles(l,n):
		divs = []
		for tup in permutations(l,3):
			i = int(''.join(tup[0:3]))
			if i % n == 0:
				divs.append([i for i in tup])
		return divs
	digs = digits()
	primes = [17]
	for p in primes:
		divs = divisibles(digs,p)
	for div in divs:
		digis = [i for i in digs if i not in div]
		print (digis)
		
		
		
def prob44():
	pents = [0]
	maxi = 2111
	mini = inf
	for i in range(1,maxi):
		pents.append(int(i*(3*i-1)/2))
	print(len(pents),'pentagonal numbers, max = ',pents[-1])
	for i, j in enumerate(pents):
		for k in pents[1:i]:
			if (j+k) in pents[1:2*i]:
				print(j-k)
				if (j-k) in pents[1:i]:
					mini=min(mini,j-k)
					print(j,k,j-k, mini)
	return mini
	
	
def prob45():
	hexn = hexagonals()
	pentn = pentagonals()
	trin = triangulars()
	for h in hexn:
#               print('h',h)
		while True:
			p = next(pentn)
#                       print(p)
			if p == h:
				while True:
					t =  next(trin)
					if t == p:
						print('winner', t)
					if t>p:
						break
			if p>h:
				break
				
				
				
def prob49():
	maxi = 10000
	prime = [True]+[True,False]*(maxi//2)
	primes = []
	for i in range(3,maxi,2):
		if prime[i]:
			if i > 999:
				primes.append(i)
			for j in range(i*2,maxi,i):
				prime[j] = False
	for i, p in enumerate(primes):
		for q in primes[i+1:]:
			r = q + (q-p)
			if r in primes:
				if sorted(str(p)) == sorted(str(q)) == sorted(str(r)):
					if p != 1487:
						return f'{p}{q}{r}'
						
						
						
def prob51():
	unhappy = True
	n = 10
	while unhappy:
		if isprime(n):
			print(n)
			for i, digit in enumerate(str(n)[:-1]):
				primes = []
				for j in digits():
					n1 = list(str(n))
					n1[i] = j
					if isprime(int(''.join(n1))):
						primes.append(''.join(n1))
				if primes[0][0] == '0':
					primes= primes[1:]
				if len(primes) >= 6:
					unhappy = False
					print(primes)
					break
			n+=1
		else:
			n+=1
			
def prob74():
	facts = {}
	chains=[]
	for i in range(10):
		facts[str(i)] = factorial(i)
	for i in range(1000000):
		chain = [i]
		new = i
		while True:
			new = sum([facts[d] for d in str(new)])
			if new in chain:
				chains.append(len(chain))
				break
			elif new < i:
				chains.append(len(chain)+chains[new])
				break
			else:
				chain.append(new)
	return (chains.count(60))
	
def prob75():
	maxperim = 1500000
	summ = Counter()
	
	for tup in euclidtriangles(maxperim):
#               print(tup)
		summ.update([(sum(tup)*i) for i in range(1, (maxperim//sum(tup))+1)])
#                       print(summ)
#       print(max(summ))
	return len([i for i, c in summ.items() if c ==1])
	
def prob92():
	def squaredigs(n):
		nonlocal nums
		n = sum([int(s)**2 for s in str(n)])
		if n ==89:
			nums[n] = 89
			return 89
		if n == 1:
			nums[n] = 1
			return 1
		if nums[n]:
			return nums[n]
		else:
			nums[n] =  squaredigs(n)
			return nums[n]
			
	maxi = 10000000
	nums = [False for i in range(maxi)]
	for i in range(1, maxi):
		if not nums[i]:
			nums[i]=squaredigs(i)
	#print([(i,n) for i,n in enumerate(nums)])
	return nums.count(89)
	
def prob111():
	n = 4 # digits long
	for d in range(10): #start with 0s
		for i in range(n):
			pass
		lnd = []
		uniques = set([])
		l = [str(d)*(n -d)]
		if n > 0:
			for b in range(10**n):
				m = l.copy()
				uniques.add(tuple(sorted([str(l[0]),('00000000'+str(b))[-i:]])))
				
			print(uniques)
			
		else:
			uniques.add(l[0])
#                               print(l,i)
			pass
		print(n,d,len(uniques))
		
def prob121():
	turns=15
	reds = 1
	blues = 1
	hands = ['']
	for i in range(turns):
		hands = [x+c for x in hands for c in 'br']
	wins = [x for x in hands if x.count('b') > turns/2]
#       path = [[x[:i] for i in range(turns)]for x in hands if x.count('b') > turns/2]
#       l=(zip(wins,path))
	probs = [prod([(1-1/(2+i),1/(2+i))[test[i]=='b'] for i in range(turns)]) for test in wins]
#       print(sum(probs))
	return int(1/sum(probs))//1
	
def prob170():
	maxi = 0
	for la in range(1,4):
		print(f'la = {la}')
		for lb in range(1,(10-la)//2):
		
			for i in pandigitals(0,9):
				i = str(i)
				a,b,c = int(i [:la]), int(i[la:la+lb]), int(i[la+lb:])
				if ispandigital(str(a)+str(b)+str(c),0,9) and ispandigital(str(a*b)+str(a*c),0,9):
					r = max(int(str(a*b)+str(a*c)),int(str(a*c)+str(a*b)))
					if r > maxi:
						maxi = r
						print(a,b,c,maxi)
						
	print('done', maxi)
	return maxi
	
def prob233():
	pass
	
	
prob = 23
answer = eval('prob'+str(prob))()
print(f'answer = {answer}')
clipboard.set(str(answer))
def web():
	webbrowser.open('https://projecteuler.net/problem='+str(prob))

