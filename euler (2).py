# Recovery Key: 1253268-dNzwKHLJwBFYfYqpHPFzzq4IsqUa5Ideuw2rL2Cg
from functools import reduce, partial
import operator
from pprint import pprint
from copy import deepcopy
import string
from itertools import permutations, combinations, tee
from collections import OrderedDict, Counter,   Hashable
from decimal import Decimal, getcontext
from fractions import gcd
import math
import time
import clipboard
import requests
from PIL import Image
import io
from random import shuffle, randint

import collections

class memoized(object):
	'''Decoratort. Caches a function's return value each time it is called.
	If called later with the same arguments, the cached value is returned
	(not reevaluated).
	'''
	def __init__(self, func):
		self.func = func
		self.cache = {}
	def __call__(self, *args):
		if not isinstance(args, Hashable):
			# uncacheable. a list, for instance.
			# better to not cache than blow up.
			return self.func(*args)
		if args in self.cache:
			return self.cache[args]
		else:
			value = self.func(*args)
			self.cache[args] = value
			return value
	def __repr__(self):
		'''Return the function's docstring.'''
		return self.func.__doc__
	def __get__(self, obj, objtype):
		'''Support instance methods.'''
		return partial(self.__call__, obj)
		
		
def digits(n : int):
	digs = []
	while n:
		digs.append(n % 10)
		n = n // 10
	return digs
		
		
def digitsum(n : int):
	summ = 0
	while n:
		summ += n % 10
		n = n // 10
	return summ
	
def digitsum_base(n:str, base:int):
	summ = 0
	s = (string.digits + string.ascii_lowercase)[:base]
	for l in n:
		summ += s.find(l)
	return summ
	
def concat_digits(n1, n2):
	return n1*10**int(math.log10(n2)+1)+n2
	
def n_in_base(n: int, base: int):
	s = (string.digits + string.ascii_lowercase)[:base]
	out = []
	while n:
		n,d = divmod(n,base)
		out.append(s[d])
	return ''.join(out[::-1])
	
def n_in_base14(n: int, s='0123456789abcd'):
	out = []
	while n:
		n,d = divmod(n,14)
		out.append(s[d])
	return ''.join(out[::-1])
	
	
def n_from_base(n: str, base: int):
	s = (string.digits + string.ascii_lowercase)[:base]
	out = 0
	unitval = 1
	for d in n[::-1]:
		out+= unitval * s.find(d.lower())
		unitval *= base
	return out

@memoized
def isprime(n):
	if n < 2:
		return False
	if n > 2 and n%2 == 0:
		return False
	else:
		for i in range(3, int(n//n**0.5)+1,2):
			if n % i == 0:
				return False
		return True
		
def factors(n):
	facts = set([1,n])
	for i in range(2, int(n**0.5)+1):
		if n%i == 0:
			facts.update((i, n//i))
	return facts
	
def iscoprime(n,m):
	nfacts = factors(n)
	nfacts.remove(1)
	mfacts = factors(m)
	for i in mfacts:
		if i in nfacts:
			return False
	return True
	
	
def isamicable(n):
	divs = properdivisors(n)
	divsum = sum(divs)
	if divsum == n:
		return False
	amisum = sum(properdivisors(divsum))
	return n == amisum
	
	
def ispandigital(n, min = 1, max=9):
	digits = string.digits[min:max+1]
	return len(str(n)) == max -min+1 and all([i in str(n) for i in digits])
	
	
def isharshad(n : int):
	return n % sum([int(l) for l in str(n)]) == 0
	
def isstrongharshad(n):
	return isharshad(n) and isprime(n // sum([int(l) for l in str(n)]))
	
def pandigitals(min=1,max=9):
	for i in permutations(string.digits[min:max+1]):
		yield int(''.join([d for d in i]))
		
		
def prod(iterable):
	return reduce(operator.mul, iterable, 1)
	
	
def factorial(n: int):
	return prod(range(1,n+1))
	
def divisors(n: int, divisorlist: set):
	divisorlist.update([1, n])
	for i in range(2,int(n**0.5)+1):
		if n % i == 0:
			divisorlist.update([i, n//i])
			if n//i not in divisorlist:
				divisors(n//i, divisorlist)
	return divisorlist
	
@memoized
def divisors2(n: int):
	divisorlist=set()
	divisorlist.update([1, n])
	for i in range(2,int(n**0.5)+1):
		if n % i == 0:
			divisorlist.update([i, n//i])
			if n//i not in divisorlist:
				divisorlist.update(divisors2(n//i))
	return divisorlist
	
@memoized
def primedivisors(n: int):
	divs = {x for x in getprimes(int(n//2)) if n % x == 0}
	if isprime(n):
		divs.add(n)
	return divs
	
	
def properdivisors(n: int):
	if n == 1:
		return [1]
	else:
		return {x for x in divisors(n,set()) if x != n}
		
def largestprimefactor(n):
	largest = None
	for i in range(2, int(n//n**0.5)+1):
		if n % i == 0:
			if isprime(i):
				largest = i
				print (largest)
	return largest
	
	
	
def counter(low, high):
	current = low
	while current <= high:
		yield current
		current += 1
		
		
def countreverse(high, low):
	current = high
	while current >= low:
		yield current
		current -= 1
		
		
def getfibonacci():
	old = 0
	current = 1
	while True:
		yield current
		current, old = (old + current, current)
		
		
def gettribonacci(maxi = 10**6):
	older = 1
	old = 1
	current = 1
	yield older
	yield old
	while current <= maxi:
		yield current
		current, old , older= (older + old + current, current, old)
		
def gettrianglenum(maxi):
	current = 2
	triangle = 1
	while triangle <= maxi:
		yield triangle
		triangle += current
		current += 1
		
		
def getsquares(maxi=1000000):
	for i in range(1,int(math.sqrt(maxi)+1)):
		yield i**2
		
def getcubes(maxi=1000000):
	for i in range(1,int((maxi**0.3333333)+1)):
		yield i**3
		
def getprimes(maxi = 1000000, two = True):
	p = [False, False]+ [False,True] * int(maxi//2)
	try:
		p[2] = True
	except:
		pass
	threshold = max(int(maxi **0.5)+1,3)
	if two:
		yield 2
	for i in range(3,threshold):
		if p[i]:
			for j in range(i**2,len(p),i):
				p[j] = False
			yield i
	for i in range(threshold, maxi + 1):
		if p[i]:
			for j in range(i**2,len(p),i):
				p[j] = False
			yield i
			
# Sieve of Eratosthenes
# Code by David Eppstein, UC Irvine, 28 Feb 2002
# http://code.activestate.com/recipes/117119/

def gen_primes():
	""" Generate an infinite sequence of prime numbers.
	"""
	# Maps composites to primes witnessing their compositeness.
	# This is memory efficient, as the sieve is not "run forward"
	# indefinitely, but only as long as required by the current
	# number being tested.
	#
	D = {}
	
	# The running integer that's checked for primeness
	q = 2
	
	while True:
		if q not in D:
			# q is a new prime.
			# Yield it and mark its first multiple that isn't
			# already marked in previous iterations
			#
			yield q
			D[q * q] = [q]
		else:
			# q is composite. D[q] is the list of primes that
			# divide it. Since we've reached q, we no longer
			# need it in the map, but we'll mark the next
			# multiples of its witnesses to prepare for larger
			# numbers
			#
			for p in D[q]:
				D.setdefault(p + q, []).append(p)
			del D[q]
			
		q += 1
		
		
def getprimestr(maxi = 1000000):
	p = [False, False]+[False,True]*(maxi//2)
	p[2] = True
	for i in range(3):
		if p[i]:
			yield str(i)
	for i in range(3,maxi):
		if p[i]:
			for j in range(2*i,len(p),i):
				p[j] = False
			yield str(i)
			
def geteuclidtriangles(maxi = 1000):
	for m in range(2,maxi):
		for n in range(1,m):
			if(m^n)&0b1:
				if iscoprime(m, n):
					a = m ** 2 - n ** 2
					b = 2 * m * n
					c = m ** 2 + n ** 2
					yield (a, b, c)
					
def getfigurates(n, maxi = 1000000):
	formula = {3:'x*(x+1)//2',
							4:'x**2',
							5:'x*(3 * x - 1)//2',
							6:'x*(2 * x - 1)',
							7:'x*(5 * x - 3)//2',
							8:'x*(3 * x - 2)'
							}
	x = 1
	while True:
		y = eval(formula[n])
		if y <= maxi:
			yield y
		else:
			break
		x+=1
					
					
def getconvergents(oneoffs : list,repeated : list):
	if len(oneoffs) > 1:
		repeated = oneoffs[1:] + repeated
	d = repeated[-1]
	n = 2* d + 1
	while repeated:
		d = n
		try:
			n = repeated.pop()+ repeated[-1]*d
		except IndexError:
			n = d
	if n:
		yield (oneoffs[0], 1)
	if len(oneoffs) > 1:
		for i in oneoffs[1:]:
			d = i * d + n
			n = i
		return (n + oneoffs[0] * d, d)
	while True:
		for i in repeated:
			d = i
			n = oneoffs[0] * d
		return (n,d)
		
		
def get_int_factors(n,ps = None):
	primelist = list(getprimes(n))
#	def build_factors()
	yield [1]
	if ps != None:
		ps = []
	for prime in getprimes(n):
		pass
		
	
def ispalindrome(s):
	s = str(s)
	if s == s[::-1]:
		return True
		
		
def ispythagtrip(a,b,c):
	return a ** 2 + b ** 2 == c  ** 2
	
	
def evenlydvisible(n, factor):
	return n%factor == 0
	
	
def walkthematrix(m: list, pathsum):
	global mini
	print(m[0][0], pathsum)
	pathsum += m[0][0]
	if pathsum > mini:
		return
	if len(m) == len(m[0]) == 1:
		if pathsum < mini:
			mini = pathsum
			print(pathsum)
		return
		
	if len(m[0])>1:
		if m[0][1] >= m[1][0]:
			walkthematrix([r[1:]for r in m], pathsum)
	if len(m) > 1:
		walkthematrix(m[1:], pathsum)
		
	if len(m[0]) > 1:
		if m[1][0] > m[0][1]:
			walkthematrix([r[1:]for r in m], pathsum)
			
			
def prob4():
	largest = 0
	for i in countreverse(999,100):
		for j in countreverse(i,100):
			if ispalindrome(i*j):
				if i*j > largest:
					largest = i*j
					print(i, j, largest)
					
					
def prob5bruteforce():
	i = 1
	while True:
		if all([evenlydvisible(i, j) for j in range(1,20)]):
			print(i,True)
			break
		i += 1
		
		
def prob6():
	sumsqu = 0
	summ = 0
	for i in range(1,101):
		sumsqu += i ** 2
		summ += i
		print(i, sumsqu, summ ** 2, abs(sumsqu - summ ** 2))
		
		
def printprimes(maxi):
	n = 2
	nbprimes = 0
	while nbprimes < maxi:
		if isprime(n):
			nbprimes +=1
			print (nbprimes, n)
		n += 1
		
		
def printprimes(maxval):
	n = 2
	sumprimes = 0
	while n < maxval:
		if isprime(n):
			sumprimes += n
			print (sumprimes, n)
		n += 1
		
def solvemaths1():
	prob = 'sorry,to,be,a,party,pooper'.split(',')
	prob= 'this,is,his,claim'.split(',')
	prob = ['solve','my','maths']
	prob = ['after','two','years']
	# 10 available digits
	assert  len(set(''.join(prob))) < 11
	
	print('='.join('+'.join(prob).rsplit('+',1)))
	letters = set()
	for word in prob:
		letters.update(word)
	print(letters)
	for perm in permutations(string.digits,len(letters)):
		cipher = str.maketrans({l:d for l,d in zip(letters,perm)})
#		print(cipher)
		summ = sum([int(word.translate(cipher)) for word in prob][:-1])
		target = int(prob[-1].translate(cipher))
		if summ == target:

			print([int(word.translate(cipher)) for word in prob])
#		print(summ)
	
def prob8():
	s= [int(x) for x in "7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450"]
	n = 13
	maximum = 0
	for i in range(0,len(s)-n):
		t = s[i:i+n]
		prods = prod(t)
		if prods > maximum:
			maximum = prods
			print (t, maximum)
	return maximum
	
	
def prob9():
	maxi = 1000
	for i in range(1,maxi):
		print(i)
		for j in range (i+1, maxi):
			for k in range(j+1, maxi):
				if ispythagtrip(i, j, k):
					if i + j + k == maxi:
						return prod([i,j,k])
						
						
def prob12():
	target = 500
	maxi = 0
	for i in gettrianglenum (50000):
		divisorlist = set()
		divisorlist = divisors2(i)
		if len(divisorlist) > maxi:
			maxi = len (divisorlist)
		if maxi>target:
			return i
			
def prob13():
	s = """37107287533902102798797998220837590246510135740250
	46376937677490009712648124896970078050417018260538
	74324986199524741059474233309513058123726617309629
	91942213363574161572522430563301811072406154908250
	23067588207539346171171980310421047513778063246676
	89261670696623633820136378418383684178734361726757
	28112879812849979408065481931592621691275889832738
	44274228917432520321923589422876796487670272189318
	47451445736001306439091167216856844588711603153276
	70386486105843025439939619828917593665686757934951
	62176457141856560629502157223196586755079324193331
	64906352462741904929101432445813822663347944758178
	92575867718337217661963751590579239728245598838407
	58203565325359399008402633568948830189458628227828
	80181199384826282014278194139940567587151170094390
	35398664372827112653829987240784473053190104293586
	86515506006295864861532075273371959191420517255829
	71693888707715466499115593487603532921714970056938
	54370070576826684624621495650076471787294438377604
	53282654108756828443191190634694037855217779295145
	36123272525000296071075082563815656710885258350721
	45876576172410976447339110607218265236877223636045
	17423706905851860660448207621209813287860733969412
	81142660418086830619328460811191061556940512689692
	51934325451728388641918047049293215058642563049483
	62467221648435076201727918039944693004732956340691
	15732444386908125794514089057706229429197107928209
	55037687525678773091862540744969844508330393682126
	18336384825330154686196124348767681297534375946515
	80386287592878490201521685554828717201219257766954
	78182833757993103614740356856449095527097864797581
	16726320100436897842553539920931837441497806860984
	48403098129077791799088218795327364475675590848030
	87086987551392711854517078544161852424320693150332
	59959406895756536782107074926966537676326235447210
	69793950679652694742597709739166693763042633987085
	41052684708299085211399427365734116182760315001271
	65378607361501080857009149939512557028198746004375
	35829035317434717326932123578154982629742552737307
	94953759765105305946966067683156574377167401875275
	88902802571733229619176668713819931811048770190271
	25267680276078003013678680992525463401061632866526
	36270218540497705585629946580636237993140746255962
	24074486908231174977792365466257246923322810917141
	91430288197103288597806669760892938638285025333403
	34413065578016127815921815005561868836468420090470
	23053081172816430487623791969842487255036638784583
	11487696932154902810424020138335124462181441773470
	63783299490636259666498587618221225225512486764533
	67720186971698544312419572409913959008952310058822
	95548255300263520781532296796249481641953868218774
	76085327132285723110424803456124867697064507995236
	37774242535411291684276865538926205024910326572967
	23701913275725675285653248258265463092207058596522
	29798860272258331913126375147341994889534765745501
	18495701454879288984856827726077713721403798879715
	38298203783031473527721580348144513491373226651381
	34829543829199918180278916522431027392251122869539
	40957953066405232632538044100059654939159879593635
	29746152185502371307642255121183693803580388584903
	41698116222072977186158236678424689157993532961922
	62467957194401269043877107275048102390895523597457
	23189706772547915061505504953922979530901129967519
	86188088225875314529584099251203829009407770775672
	11306739708304724483816533873502340845647058077308
	82959174767140363198008187129011875491310547126581
	97623331044818386269515456334926366572897563400500
	42846280183517070527831839425882145521227251250327
	55121603546981200581762165212827652751691296897789
	32238195734329339946437501907836945765883352399886
	75506164965184775180738168837861091527357929701337
	62177842752192623401942399639168044983993173312731
	32924185707147349566916674687634660915035914677504
	99518671430235219628894890102423325116913619626622
	73267460800591547471830798392868535206946944540724
	76841822524674417161514036427982273348055556214818
	97142617910342598647204516893989422179826088076852
	87783646182799346313767754307809363333018982642090
	10848802521674670883215120185883543223812876952786
	71329612474782464538636993009049310363619763878039
	62184073572399794223406235393808339651327408011116
	66627891981488087797941876876144230030984490851411
	60661826293682836764744779239180335110989069790714
	85786944089552990653640447425576083659976645795096
	66024396409905389607120198219976047599490197230297
	64913982680032973156037120041377903785566085089252
	16730939319872750275468906903707539413042652315011
	94809377245048795150954100921645863754710598436791
	78639167021187492431995700641917969777599028300699
	15368713711936614952811305876380278410754449733078
	40789923115535562561142322423255033685442488917353
	44889911501440648020369068063960672322193204149535
	41503128880339536053299340368006977710650566631954
	81234880673210146739058568557934581403627822703280
	82616570773948327592232845941706525094512325230608
	22918802058777319719839450180888072429661980811197
	77158542502016545090413245809786882778948721859617
	72107838435069186155435662884062257473692284509516
	20849603980134001723930671666823555245252804609722
	53503534226472524250874054075591789781264330331690"""
	t= sum([int(txt) for txt in s.split("\n")])
	print(t)
	
	
def prob18():
	s = """75
	95 64
	17 47 82
	18 35 87 10
	20 04 82 47 65
	19 01 23 75 03 34
	88 02 77 73 07 63 67
	99 65 04 28 06 16 70 92
	41 41 26 56 83 40 80 70 33
	41 48 72 33 47 32 37 16 94 29
	53 71 44 65 25 43 91 52 97 51 14
	70 11 33 28 77 73 17 78 39 68 17 57
	91 71 52 38 17 14 91 43 58 50 27 29 48
	63 66 04 68 89 53 67 30 73 16 69 87 40 31
	04 62 98 27 23 09 70 98 73 93 38 53 60 04 23"""
	m = [[int(u) for u in t.split(' ')] for t in s.splitlines(False)]
#    m = [[3], [7,4], [2, 4, 6],[8, 5, 9, 3]]
	sums = [[None for j in range(i+1)] for i in range (len(m))]
	for i, row in enumerate(sums):
		row[0] = sum([r[0] for r in m[0:i+1]])
		if i >0:
			row[-1] = sum([r[-1] for r in m[0:i+1]])
		if i > 1:
			for k,c in enumerate(row[1:][:-1]):
				sums[i][k+1] = max(sums[i-1][k], sums[i-1][k+1])+m[i][k+1]
	print(max(sums[-1]))
	
def prob21():
	summ = 0
	for i in range(1,10001):
		if isamicable(i):
			summ += i
			print(i, summ)
	return summ
	
def prob23():
	def abundantsum(i, abundants):
		for j in abundants:
			if i-j in abundants:
				return True
		return False
		
	summ = 0
	abundants = set()
	for i in range(1,28123):
		if sum(properdivisors(i)) > i:
			abundants.add(i)
		if abundantsum(i, abundants):
			continue
		else:
			summ+=i
	return summ
	
def prob24():
	i = 0
	result = ''
	for j in permutations(string.digits, len(string.digits)):
		i += 1
		if i == 1000000:
			print (i, result.join(j))
			
			
def prob27():
	maxnum = 40
	for a in range(-65,1001):
		for b in range(-999,1001,2):
			i = 0
			while isprime(i**2 + a*i + b):
				i += 1
			if i > maxnum:
				maxnum = i
				answer = a * b
				print(f"{i} primes for a ={a}, b ={b}")

	return answer
					
def prob31():
	@memoized
	def makesum(n):
		combos = set()
		coins = [200,100,50,20,10,5,2,1]
		coinstr = {200:'a',100:'b',50:'c',20:'d',10:'e',5:'f',2:'g',1:'h'}
		for c in coins:
			if c <= n:
				if n-c == 0:
					combos.add(coinstr[c])
				else:
					mcombos.update((''.join(sorted(x+coinstr[c])) for x in makesum(n-c)))
		return combos
		
	def addsum(target):
		ways = [0 for i in range(target + 1)]
		ways[0] = 1
		for i in [1,2,5,10,20,50,100,200]:
		  for j in range(i, target+1-i):
		    ways[j] += ways[j - i]
		  ways[target] += ways[target - i]
		return ways[target]
	return addsum(200)
	
	
def prob34():
	summ =  0
	for i in range(3,99999999999999):
		if i == sum([factorial(int(n)) for n in str(i)]):
			summ += i
			print(i, summ)
			
			
def prob36():
	summ = 0
	for i in range(1000000):
		if ispalindrome(str(i)):
			if ispalindrome(str(bin(i)[2:])):
				summ += i
				print(i, bin(i))
	return summ
	
	
def prob38():
	maxi = 0
	for n in range(2):
		pass
		
def prob41():
	'''if the digits of a number sum to a multiple of 3 then the number is a multiple of 3, ie not prime, so only 1,4 or 7-digit pandigital numbers can be prime (but 1 is not prime by definition)
	'''
	for perm in permutations('7654321', 7):
		if isprime(int(''.join(perm))):
			return int(''.join(perm))
#		print(int(''.join(perm)))
		
		
def prob43():
	def adddigit(l,div):
		nonlocal s
		m=list()
		for st in l:
			m.extend([d + st for d in s if int(d+st[:2]) % div == 0 and d not in st])
		return m
	s = string.digits
	divs = list(getprimes(18))
	l = permutations(s,2)
	strs = [''.join(tup) for tup in list(l)]
	while divs:
		strs = adddigit(strs,divs.pop())
		print(len(strs))
	strs = [int(d+st) for st in strs for d in s if d not in st]
	print(strs)
	return sum(strs)
	
def prob45():
	return 1533776805
	
def prob46():
	print('\nGoldbach\'s other conjecture')
	maxi = 1000000
	sq2 = [2 * x for x in getsquares(maxi/2)]
	pr = list(getprimes(maxi))
	prset = set(pr)
	vals = iter(range(7,maxi,2))
	for i in vals:
		if i not in prset:
			for s2 in sq2:
				if s2> i:
					return i
				if (i-s2) in prset:
					break
	#                               next(vals)
	
def prob47():
	target = 4
	count = 0
	answer = 1
	while True:
		if len(primedivisors(answer)) == target:
			count += 1
			if count == target:
				return (answer - target +1)
		else:
			count = 0
		answer += 1
		
def prob51():
	def checkfamily(p,pattern,n):
		family = [p+i*pattern for i in range(10)]
		digits = str(pattern)
	patterns = [int(str(bin(i))[2:]) for i in range(2,64,2)]
	print(patterns)
	primes = set([x for x in getprimes(1000000) if x > 100000])
	l = ((p,x,sum([(p+ i * x) in primes for i in range(10) if (p+i*x) < 1000000]) ) for p in primes for x in patterns)
	s = ((i,j,k) for i,j,k in l if k >7)
	for x in s:
		try:
			print(x[0],x[1])
		except :
			print(x)
			
			
			
def prob54():
	#no straight or royal flushes
	p1wins = 0
	draws = 0
	def value(card):
		vals = {'T':10,'J':11,'Q':12,'K':13,'A':14}
		try:
			return vals[card]
		except KeyError:
			return int(card)
	def rank(r):
		ranks = {'':0,'dus':1,'dusdus':2,'trips':3,'straight':4,'flush':5,'dustrips':6,'four':7,'straightflush':8}
		return ranks[r]
		
	def evalu(vals, suits, h):
		r=''
		vallist = [value(c[0]) for c in h]
		if len(vals) <= 3:
	#               print('three', vals)
			for c in vals:
		#               print(vallist.count(c))
				if vallist.count(c) == 2:
					r='dus'+r
				if vallist.count(c) == 3:
					r+='trips'
				if vallist.count(c) == 4:
					r+='four'
			return r
		if len(vals) == 4:
			return 'dus'
		if len(vals) == 5:
			if max(vals)-min(vals) == 4:
				r += 'straight'
			if len(suits) == 1:
				r += 'flush'
	#                       print('flush', r1)
		return r
		
	def p1win(h1,h2):
		vallist1 = [value(c[0]) for c in h1]
		vallist2 = [value(c[0]) for c in h2]
		vallist1 = sorted(vallist1,key = lambda x: (vallist1.count(x),x),reverse=True)
		vallist2 = sorted(vallist2,key = lambda x: (vallist2.count(x),x),reverse=True)
		for c1, c2 in zip(vallist1,vallist2):
			if value(c1) != value(c2):
				#print(vallist1,vallist2,value(c1)>value(c2))
				return value(c1)>value(c2)
		raise ValueError
	r = requests.get('https://projecteuler.net/project/resources/p054_poker.txt')
	for round in r.text.split('\n')[:-1]:
		hands = round.split(' ')
		h1 = hands[:5]
		h2 = hands[5:]
		suits1 = set(c[1] for c in h1)
		suits2 = set(c[1] for c in h2)
		vals1 = set(value(c[0]) for c in h1)
		vals2 = set(value(c[0]) for c in h2)
#               print(suits1,suits2)
		r1 = rank(evalu(vals1, suits1, h1))
		
		r2 = rank(evalu(vals2, suits2, h2))
		
#               if r1 or r2:
	#               print(h1, r1,rank(r1), h2,r2 ,rank(r2))
		if r1 > r2:
			p1wins += 1
		if r1 == r2:
			if p1win(h1,h2):
				p1wins += 1
	print(f'Player one wins {p1wins} and draws {draws}')
	return p1wins
	
	
	
def prob55():
	lychrels = []
	for i in range(1,10000):
		ia = i
		for j in range(50):
			i = i+int(str(i)[::-1])
			if ispalindrome(i):
				lychrels.append(ia)
				break
	print(len(lychrels), lychrels)
	
def prob57():
	terms = [1]
	answer = 0
	for i in range(1000):
		term2 = 2
		terms += [term2]
	n = 2
	d = 1
	for i in range(len(terms)):
		n, d = terms.pop()*n + d, n
		if int(math.log10(n-d))>int(math.log10(d)):
			answer+=1
#		print(n - d, d)
	return answer
	
	
def prob58():
	layer = 1
	n0 = 1
	diags = 1
	diagprimes = 0
	while True:
		layer += 1
		n1 = ((2 * layer) - 1) ** 2
		for i in [n1 - j*((n1- n0)//4) for j in range(1,4)]:
			if isprime(i):
				diagprimes += 1
		diags += 4
#		print(layer, diagprimes/diags)
		if diagprimes/diags < 0.1:
			return 2 * layer - 1
		n0 = n1
	
	
def prob60():
	def store(j, k):
		nonlocal primes
		j, k = min(j,k), max(j, k)
		try:
			primes[j].add(k)
		except KeyError:
			primes[j]=set((k,))
			
	primes = {}
	for q in getprimes(200000):
		for i in range(int(math.log10(q))):
			j=q//10**(i+1)
			k=q%10**(i+1)
			kj = concat_digits(k,j)

			if isprime(j) and isprime(k) and isprime(kj) and int(math.log10(q)) == int(math.log10(kj)):
	#			print(q,i,j,k,kj)
				store(j,k)
	print(len(primes))
	target = 4
	fours = []
	answer = math.inf
	for root in sorted(primes.keys()):
		print(root,primes[root])
		if len(primes[root]) > target - 1:
			for combi in combinations(primes[root],target - 1):
				if True: #sum(combi) + root < answer:
					for perm in permutations(combi,2):
						k,j = perm
						kj = concat_digits(k,j)
	#					print(k,j,kj)
						if not isprime(kj):
							break
					else:
						print(root, combi)
						fours.append((root,)+combi)
	print(fours)
	
	for p in getprimes():
		if p < answer:
			for four in fours:
				if p not in fours:
					if all((isprime(concat_digits(p,f)) and isprime(concat_digits(f,p)) for f in four)):
						if sum(four) + p < answer:
							answer = sum(four) + p
							print(p,four)
	return (answer)

	
def prob61():
	def addfig(liste: list,reste: list):
		if sum(reste) == 0 and len(liste)== 6:
			if liste[0]//100 == liste[-1] % 100:
				print(liste, sum(liste))
				clipboard.set(str(sum(liste)))
				return sum(liste)
		for i, n in enumerate(reste):
			if n:
				for fig in figs[i]:
					assert sum(reste) + len(liste) == 6
#					print(fig)
					if fig//100 == liste[-1] %100 and fig not in liste:
						reste[i] = False
#						print(liste+[fig],i)
						addfig(liste+[fig], list(reste))
						reste[i] = True

	# from 0:7 to 4:3						
	figs = [[i for i in getfigurates(7-j,9999)  if i > 999] for j in range(5)]	
	#print(figs)
	for fig in (i for i in getfigurates(8,9999) if i > 999):
		remaining = [True]*5
#		print ('8:',fig,remaining)
		addfig([fig],remaining)
	
def prob62():
	c = Counter()
	for i in getcubes(1e12):
		c[str(sorted(str(i)))]+=1
		if c.most_common(1)[0][1] == 5:
			targetset = str(sorted(str(i)))
			print ('found the set for', i, targetset)
			break
	for i in getcubes(1e12):
		if str(sorted(str(i))) == targetset:
			return i
			
def prob64(): 
	i = 23
	entier = int(i**0.5)
	n = entier
	while True:
		d = i - n ** 2
		entier, n = int((entier - d) / d) +1, entier - d
		print(entier,n,d)
		
			
			
def prob65():
	terms = [2]
	for i in range(33):
		term2 = 2 * (i + 1)
		terms += [1, term2, 1]
#	print(terms)
	terms.pop()
#	terms.reverse()
	n = d = 1
	for i in range(99):
		n, d = terms.pop()*n + d, n
	#	print(n,d)
	return digitsum(n)
			
def prob66():
	maxisq = 0
	maxii = None
	for i in range(1,1000):
		answered = False
		if math.sqrt(i) % 1 == 0:
			continue
		for x2 in getsquares(10000000000000):
			if x2 == 1:
				continue
			if math.sqrt((x2 - 1) / i ) % 1 == 0:
				answered= True
				if x2 > maxisq:
					maxisq = x2
					maxii = i
					print(i, math.sqrt(x2), (math.sqrt(x2)+1)/i)
				break
		if not answered:
			print(i,'failed')
	return maxii
	
			
def prob67():
	with open('p067_triangle.txt') as f:
		m = [[int(y) for y in x.strip("\n").split(" ")] for x in f.readlines()]
		
	sums = [[None for j in range(i+1)] for i in range (len(m))]
	for i, row in enumerate(sums):
		row[0] = sum([r[0] for r in m[0:i+1]])
		if i >0:
			row[-1] = sum([r[-1] for r in m[0:i+1]])
		if i > 1:
			for k,c in enumerate(row[1:][:-1]):
				sums[i][k+1] = max(sums[i-1][k], sums[i-1][k+1])+m[i][k+1]
				
	print(max(sums[-1]))
	
	
def prob70():
	target = 10 **7
	totientmin = math.inf
	for i in getprimes(target//2):
		for j in getprimes(min(target // i, i)):
			totient = (i - 1) * (j - 1)
			if i * j / totient < totientmin:
				if sorted(digits(totient)) == sorted(digits(i*j)):
					totientmin = i * j / totient
					answer = i * j
	return answer

def prob71():
	target = 3/7
	min_diff = 1.
	for d in range(1000000,0,-1):
		n = d // (7/3)
		diff = target - (n/d)
		if diff < min_diff:
			min_diff = diff
			min_n = n
			min_d = d
	print(min_n, min_d)
	return int(min_n)
	
	
def prob72():
	answer = 0
	target = 10**6
	primedivs = {i:[] for i in range(target + 1)}
	for p in getprimes(target+1):
		for j in range(p,target+1, p):
			primedivs[j].append(p)
	for i in range(2,target+1):
		answer += i* prod( [(1-1/p) for p in primedivs[i]])
#		print(i, primedivisors(i),answer)
	return int(answer) 
	
	
def prob73():
	frac1 = 1/3
	frac2 = 1/2
	answer = 0
	for d in range(5,12001):
		divisorset=set()
		for i in range(2,int(d**0.5)+1):
			if d % i == 0:
#				divisorset.update([i, d//i])
				for j in range(i, d, i):
					divisorset.add(j)
				for j in range(d //i, d, d // i):
					divisorset.add(j)
#		print(d, divisorset)
		n1 = int(d // (1 / frac1)) + 1
		n2 = int(d // (1 / frac2))
		for n in range(n1,n2+1):
			if n not in divisorset:
				answer += 1
		#		print(n,d,n/d)
	return answer
	
def prob75():
	maximum = 1500000
	m = int((maximum//2)** 0.5)
	lengths = set()
	multilengths = set()
	for tri in geteuclidtriangles(m):
		length = sum(tri)
		if length <=maximum:
			for l in range(length,maximum+1, length):
				if l in lengths:
					multilengths.add(l)
				lengths.add(l)
	#			print(l)
			
#			print(tri)
	return len(lengths) - len(multilengths)
	
	
def prob76():
	@memoized
	def routes(total, maxval):
		if any((total == 0, total == 1, maxval == 1)):
			return 1
		way = 0
		for i in range(min(maxval, total), 0, -1):
			way += routes(total - i, i)
		return way
			
	target = 100
	ways = {1 : 0, 2: 1}
	way = 0
	for i in range(target-1, 0, -1):
		way += routes(target-i, i)
#		print(i,way)

	return way
	
	
def prob77():
	def sumways(n, l):
		if n == 0:
#			print('done')
			return 1
		ways = 0
		nonlocal primes
		ps = [i for i in primes if i <= min(l, n)]
		for p in ps:
#			print(n, p)
			ways += sumways(n - p, p)
		return ways
	
	target = 71
	primes = list(getprimes(target))
	primes.reverse()
	return  sumways(target, target)
	
def prob78():
	@memoized
	def routes(total, maxval):
		if any((total == 0, total == 1, maxval == 1)):
			return 1
		way = 0
		for i in range(min(maxval, total), 0, -1):
			way += routes(total - i, i)
		return way % 1000000
	
	def sumways(total):
		ways = [0 for i in range(total+1)]
		ways[0] = 1
		for i in range(1, total+1):
			for j in range(i,total+1):
				ways[j] += ways[j- i]
#			if total - i < i:
#				ways[i] += ways[0]
#			ways[total] += ways[total - i]
			if ways[i] % 10 **4 == 0:
				print(i, ways[i])
	
	target = 10000
	sumways(target)
''''	while True:
		piles = 0
		for i in range(target+1):
			piles += routes(target-i,i)
		if piles % 1000 == 0:
			print(target, piles)
			if piles % 1000000 == 0:
				print(target, piles)
				return target
		target+= 1'''
	
def prob80():
	getcontext().prec = 102
	answer = 0
	for i in range(2, 100):
		if math.sqrt(i) % 1 == 0:
			continue
		else:
			summ = sum([d for d in Decimal(i).sqrt().as_tuple()[1][:100]])
			answer += summ
#			print(i, summ)
	return answer
	
	
def prob81():
	with open("p081_matrix.txt") as f:
		m = [[int(y) for y in x.strip("\n").split(",")] for x in f.readlines()]
	print(m[0])
	print(m[1])
	print(m[2])
	print(m[3])
	#   matrix = [[131,673,234,103,18], [201,96,342,965,150], [630,803,746,422,111],[537,699,497,121,956], [805,732,524,37,331]]
	#initialise values on the top and left hand rows
	sums = [[None for j in range(len(m))] for i in range (len(m))]
	sums[0] = [sum(m[0][0:i+1]) for i in range(len(m[0]))]
	for i in range(1,len(m)):
		sums[i][0] = sum([m[j][0] for j in range(i+1)])
		for j in range(len(m)-1):
			sums[i][j+1] = min(sums[i][j], sums[i-1][j+1])+m[i][j+1]
	for i, row in enumerate(m):
		pass
	print(sums[0])
	print(sums[1])
	print(sums[2])
	print(sums[-1][-1])
	
	
def prob82():
	with open("p082_matrix.txt") as f:
		m = [[int(y) for y in x.strip("\n").split(",")] for x in f.readlines()]
	print(m[0])
	print(m[1])
	print(m[2])
	print(m[3])
	sums = [[None for j in range(len(m))] for i in range(len(m))]
	for i in range(len(m)):
		sums[i][0] = m[i][0]
	for c in range(1,len(m)):
		for r in range(len(m)): # step right and down
			if r == 0:
				sums[r][c] = sums[r][c - 1] + m[r][c]
			else:
				sums[r][c] = min(sums[r-1][c]+ m[r][c], sums[r][c-1]+ m[r][c])
		for r in range(len(m)-2,-1,-1): # step back up
			sums[r][c] = min(sums[r - 1][c] + m[r][c], sums[r][c - 1] + m[r][c], sums[r + 1][c] + m[r][c])
			
	print('sums')
	print(sums[0])
	print(sums[1])
	print(sums[2])
	print(min([s[-1] for s in sums]))
	
def prob83():
	def neighbours(r, c, m):
		l = []
		for i, j in [(r+1, c),(r-1, c),(r, c+1),(r, c-1)]:
			if all([i>=0, j>=0, i<len(m), j<len(m[0])]):
				l.append(m[i][j])
		return l
		
	with open("p083_matrix.txt") as f:
		m = [[int(y) for y in x.strip("\n").split(",")] for x in f.readlines()]
	print(m[0])
	print(m[1])
	print(m[2])
	print(m[3])
	sums = [[math.inf for j in range(len(m))] for i in range(len(m))]
	for i in range(len(m)):
		sums[i][0] = sum([m[j][0] for j in range(i+1)])
		
	for c in range(len(m)): # step right ...
		for r in range(len(m)): # ... and down
			if r == 0 and c== 0:
				sums[r][c] =  m[r][c]
			else:
				print(neighbours(r,c,sums))
				sums[r][c] = min(neighbours(r,c,sums))+ m[r][c]
		for r in range(len(m)-1,0,-1): # step back up
			sums[r][c] = min(neighbours(r,c,sums)) + m[r][c]
			
	for i in range(10):
		for c in range(len(m)-1,0,-1): # step left from the last column...
			for r in range(1,len(m),-1): # ... and down - cant go left along 1st and last row
				sums[r][c] = min(neighbours(r,c,sums))+ m[r][c]
			for r in range(len(m)-1,0,-1): # step back up
				sums[r][c] = min(neighbours(r,c,sums))+ m[r][c]
		print(sums[-1][-1])
		
		for c in range(len(m)): # step right ...
			for r in range(len(m)): # ... and down
				if r == 0 and c== 0:
					sums[r][c] =  m[r][c]
				else:
					sums[r][c] = min(neighbours(r,c,sums))+ m[r][c]
			for r in range(len(m)-1,0,-1): # step back up
				sums[r][c] = min(neighbours(r,c,sums)) + m[r][c]
				if r == 0 and c == 0:
					sums[r][c] = m[r][c]
					
	print('sums')
	print(sums[0])
	print(sums[1])
	print(sums[2])
	print(sums[-1][-1])
	
	
def prob84():
	print('let\'s play')
	answer = None 
	squares = [0]*40
	square = 0
	double = 0
	round = 0
	while True:
		a = 0
		b = 0
		double = 0
		while a == b:
			a = randint(1,4)
			b = randint(1,4)
			square = (square + a + b ) % 40
			if a == b:
				double += 1
			if double == 3:
				double = 0
				square = 10
				break
		if square == 30:
			square = 10
		if square in (2,17,33):
			cc = randint(1,16)
			if cc == 1:
				square = 0
			elif cc == 2:
				square = 10
		if square in (7, 22,36):
			ch = randint(1,16)
			chance = {1:0,2:10,3:11,4:24,5:39,6:5}
			if ch in chance:
				square = chance[ch]
			elif ch in (7,8):
				square = (((square +5) %10) + 5) %40
			elif ch == 10:
				square -= 3
			elif ch == 9:
				if square in (7,36):
					square = 12
				elif square == 22:
					square = 28
		squares[square] += 1
		round += 1
		if round % 100000 == 0:
#			pprint(list(enumerate(squares)))
			return int(''.join([str(squares.index(x)) for x in sorted(squares)[-3:]][::-1]))
			
			
def prob86():
	target = 2000
	for tri in geteuclidtriangles():
		pass
	answer = None 
	return answer
	
def prob87():
	target = 50000000
	can = [False] * (target)
	for k in getprimes(int(target ** 0.25)):
		k = k ** 4
		for j in getprimes(int((target - k) ** 0.3333)):
			j = j **3
			for i in getprimes(int((target - k - j) ** 0.5)):
				i = i **2
				can[i+j+k] = True
#				print(i,j,k)
#	print([i for i in range(target) if can[i]])
	return len([i for i in range(target) if can[i]])
	
def prob89():
	with open("p089_roman.txt") as f:
		numerals = [x.strip('\n') for x in f.readlines()]
	baselength = sum(len(x) for x in numerals)
	values = []
	shortnums = []
	lettervals = {
	"M" : 1000,
	"CM" : 900,
	"D" : 500,
	"CD" : 400,
	"C" : 100,
	"XC" : 90,
	"L" : 50,
	"XL" : 40,
	"X" : 10,
	"IX" : 9,
	"V": 5,
	"IV" : 4,
	"I" : 1}
#    lettervals = OrderedDict(lettervals, key = lambda x : x.items())
	def short(v):
		return(v // 1000 * r'M' +
		v % 1000 // 900 * r'CM' +
		v % 1000 % 900 // 500 * r'D' +
		v % 1000 % 900 % 500 // 400 * r'CD' +
		v % 500 % 400 // 100 * r'C' +
		v % 100 // 90 * r'XC' +
		v % 100 % 90 // 50 * r'L' +
		v % 100 % 90 % 50 // 40 * r'XL' +
		v % 50 % 40 // 10 * r'X' +
		v % 10 // 9 * r'IX' +
		v % 10 % 9 // 5 * r'V' +
		v % 10 % 9 % 5 // 4 * r'IV' +
		v % 5 % 4 // 1 * r'I')
		
	def value(num):
		v = 0
		while num:
#            print(num, v)
			if len(num) == 0:
				break
			if len(num) > 1:
				if num[:2] in lettervals.keys():
					v += lettervals[num[:2]]
					try:
						num = num[2:]
					except:
						num = ""
				else:
					v += lettervals[num[0]]
					try:
						num = num[1:]
					except:
						num = ""
			else:
				v += lettervals[num[0]]
				try:
					num = num[1:]
				except:
					num = ""
		return v
		
	for num in numerals:
		print(num)
		v = value(num)
		values.append(v)
		shortnum = short(v)
		print(v, shortnum)
		shortnums.append(shortnum)
		if value(shortnum) != v:
			print('error', num, v, shortnum, value(shortnum))
	print(baselength, sum([len(x) for x in shortnums]))
	
def prob95():
	answer = 0
	maxchain = 0
	target = 10 ** 6
	tested = {i for i in getprimes(target)}
	tested.add(1)
	for i in range(12494,target):
#		print(i)
		if i in tested:
			pass
		else:
			factsum = sum(properdivisors(i))
			chain = [i]
			while True:
#				print(factsum)
				chain.append(factsum)
				oldfs= factsum
				factsum = sum(properdivisors(factsum))
				if factsum in tested or factsum > target or factsum == oldfs:
					tested.update(chain)

					break
				elif factsum in chain:
					if factsum == i:
						if len(chain) > maxchain:
							answer = min(chain)
							maxchain = len(chain)
							print(answer, chain)
							tested.update(chain)
					break
#			print(chain)

	return answer
	
	
def prob96():
	def filter():
		nonlocal rows, missing, cells, columns
		for r in range(9):
			for c in range(9):
				cells[r//3*3+c//3].add(rows[r][c])
				for x in set(rows[r]).union(columns[c].union(cells[r//3*3+c//3])):
					x = int(x)
					try:
						missing[r][c].remove(x)
#							print(x, missing[r][c])
						if len(missing[r][c]) == 1:
							rows[r][c] =str(missing[r][c][0])

							print('@@@@@@@@@@@@@@&@&@______________deduced one value')
					except ValueError:
						pass

					if len(missing[r][c]) == 2:
						if missing[r].count(missing[r][c]) == 2:
							(a,b) = [x for x in missing[r][c]]
							for cell in missing[r]:
								if cell != missing[r][c]:
									try:
										cell.remove(a)
										print('found row pair')
									except ValueError:
										pass
									try:
										cell.remove(b)
										print('found row pair')
									except ValueError:
										pass
										
					if len(missing[r][c]) == 2:
						col = [missing[i][c] for i in range(9)]
						if col.count(missing[r][c]) == 2:
							(a,b) = [x for x in missing[r][c]]
							for cell in col:
								if cell != missing[r][c]:
									try:
										cell.remove(a)
										print('found col pair')
									except ValueError:
										pass
									try:
										cell.remove(b)
										print('found col pair')
									except ValueError:
										pass
						
	def findsolos():
		nonlocal rows, missing, cells, columns
		for i in range(9):
			for value in range(1,10):
				miss_row = missing[i]
				miss_col = [x[i] for x in missing]
				if [x for m in miss_row for x in m].count(value) == 1:
					for j, m in enumerate(miss_row):
						if value in m and rows[i][j] and str(value) not in rows[i]+[rows[k][j] for k in range(9)] == '0':
							rows[i][j] = str(value)
							print('rows',i,j,value)
						
				if [x for m in miss_col for x in m].count(value) == 1:
					for j, m in enumerate(miss_col):
						if value in m and rows[j][i] and str(value) not in rows[j]+[rows[k][i] for k in range(9)]== '0':
							rows[j][i] = str(value)
							print('cols',j,i,value)
							
	def solve(rows, complete):
		...
	
	def fillin(n, test = None):
		if n== 81:
			print(test)
			return test
		row = n // 9
		column = n % 9
		nonlocal rows, missing, cells, columns
		gridsum = sum([int(i) for row in rows for i in row])
		gridnum = sum([1 for row in rows for i in row if i != '0'])
		if test == None:
			test = [[None for x in row ] for row in rows]
		if rows[row][column] == '0':
			for i in range(1,10):
				if str(i) in set(rows[row]).union(columns[column]).union(set(test[row][:column+1])).union(set(([test[j][column] for j in range(row+1)]))):
					continue
				else:
					test[row][column]  = str(i)
					testsum = sum([int(j) for row in test for j in row if j != None])
					testnum = sum([1 for row in test for j in row if j != None])
					if testsum + gridsum == 45 * 9 and testnum + gridnum == 81:
						return test
					else:
						test = fillin(n + 1, test)
						testsum = sum([int(j) for row in test for j in row if j != None])
						testnum = sum([1 for row in test for j in row if j != None])
						if testsum + gridsum == 45 * 9 and testnum + gridnum == 81:
							return test
					
		else:
			test = fillin(n+1, test)
			testsum = sum([int(j) for row in test for j in row if j != None])
			testnum = sum([1 for row in test for j in row if j != None])
			if testsum + gridsum == 45 * 9 and testnum + gridnum == 81:
				return test
		
		return test
					
		
	sudoku_url = "https://projecteuler.net/project/resources/p096_sudoku.txt"
	solved = 0
	r = (i for i in requests.get(sudoku_url).text.splitlines())
	while r:
		try:
			print (next(r))
		except StopIteration:
			break
		rows = []
		cells = []
		resolu = False
		for i in range(9):
			rows.append(list(next(r)))
			cells.append(set())
			print(''.join(rows[-1]))
		missing = [[list(range(1,10)) if x == '0' else [] for x in row ] for row in rows]
		oldgridsum = 0
		failed = 0
		for i in range(20):
			columns = [set([rows[c][r] for c in range(9)]) -set('0') for r in range(9)]
			
			'''		print('rows')
			for row in rows:
				print(''.join(row))
			print ('columns')
			for column in columns:
				print(''.join(column))
			for i, cell in enumerate(cells):
				print (i, ''.join(cell))'''
				
			filter()
			findsolos()
			gridsum = sum([int(i) for row in rows for i in row])
			if gridsum == 45 *9:
				print('SOLVED by deduction!!!!!!!!!!!!!', rows[0][0:3])
				resolu = True
				print('\n'.join([''.join([x for x in row])for row in rows]))
				solved +=1
				break
			if gridsum == oldgridsum:
				failed += 1
				if failed == 3:
					print('failed by deduction')
					print('\n'.join([''.join([x for x in row])for row in rows]))
					break
			oldgridsum = gridsum
		
		if not resolu:
			print('trying by brute force')
			fills = fillin(0)
			print(fills)
			for m,row in enumerate(fills):
				for n, val in enumerate(row):
					if val:
						rows[m][n] = val
			gridsum = sum([int(i) for row in rows for i in row])
			
			if gridsum == 45 *9:
				print('SOLVED by brute force!!!!!!!!!!!!!', rows[0][0:3])
				print('\n'.join([''.join([x for x in row])for row in rows]))
				solved +=1
		
#		for line in (''.join(str(x)) for y in [rows, columns, missing] for x in y):
	#		print(line)
	print(solved, 'of 50 grids solved')
		
def prob96():
	sudoku_url = "https://projecteuler.net/project/resources/p096_sudoku.txt"
	solved = 0
	answer = 0
	r = (i for i in requests.get(sudoku_url).text.splitlines())
	
	def solvegrid(grid, i):
		if i == 81:
			if '0' in grid:
				return grid, False
			else:
				return grid, True
		if grid[i] == '0':
			row,col = divmod(i,9)
			boxstart = ((row//3)*3*9)+(col//3)*3
			neighbours = set()
			neighbours.update(grid[row*9:(row+1)*9])
			neighbours.update(grid[col:82:9])
			neighbours.update(grid[boxstart:boxstart+3]
			+grid[boxstart+9:boxstart+12]
			+grid[boxstart+18:boxstart+21])
			neighbours.remove('0')
	#			if row == col:
	#				print(row,col,neighbours)
			missing = set('123456789').difference(neighbours)
			for val in missing:
				newgrid = grid.copy()
				newgrid[i] = val
				newgrid, done = solvegrid(newgrid,i+1)
				if done:
					return newgrid, True
		else:
			grid, done = solvegrid(grid,i+1)
			if done:
				return grid, True
		return grid, False
	
	while r:
		try:
			print (next(r))
		except StopIteration:
			break
		rows = []
		cells = []
		resolu = False
		grid = []
		for i in range(9):
			grid.extend(next(r))
		grid , done = solvegrid(grid,0)
		print('\n'.join(''.join(grid[i:i+9]) for i in range(0,82,9)))
		if done:
			solved += 1
			print('solved!!!', solved)
			answer += int(''.join(grid[:3]))
	return answer

	
def prob97():
	'''28433×2**7830457+1 =
	= 28433 * 2 * 2**7830456+1
	= 28433 * 2 * 2**24 *326269  +1
	'''
	ans = 28433*2 *2**24
	for i in range(326268):
		ans = (ans%10**10)*2**24
	return str(ans+1)[-10:]
	
def prob98(): 
	answers = []
	url = 'https://projecteuler.net/project/resources/p098_words.txt'
	words = requests.get(url).text.replace('"','').split(',')
#	print(words)
	longest = max([len(word) for word in words])
#	squares = list(getsquares(10**longest))
	print('longest word length',longest)
	lengths = iter(range(longest,1,-1))
	for length in lengths:
		lenwords = [word for word in words if len(set(word)) == length]
		letters = [sorted(w) for w in lenwords]
		print('\nNumber of unique letters',length)
		anagrams = []
		for word in lenwords:
			if letters.count(sorted(word)) >1:
				anagrams.append(word)
		print(len(anagrams), 'anagrams')
#		print(anagrams)
		if anagrams:
			lensquares = list((str(sq) for sq in getsquares(10**length) if int(math.log10(sq)) == length -1))
			for word1 in anagrams:
				for word2 in [w for w in anagrams if w != word1 and sorted(w) == sorted(word1)]:
					mapping = {l1:l2 for (l1,l2) in zip(word1,word2)}
					for sq1 in lensquares:
						if len(set(sq1)) == len(set(word1)):
							sq2 = ''.join([sq1[word1.index(mapping[l])] for l in word1])
							if sq2 in lensquares:
								if sorted(sq2) == sorted(sq1):
									print('yes', word1,word2,sq1,sq2)
									answers.append(max(int(sq1), int(sq2)))
		if answers:
			return max(answers)

					
	#				print(word1,word2)
			print(len(lensquares), 'squares')
	
	
	
def prob99():
	file_url = "https://projecteuler.net/project/resources/p099_base_exp.txt"
	tuplist = [eval(i) for i in requests.get(file_url).text.splitlines()]
	candidates = list(tuplist)
	out = set()
	for prime in reversed([1,]+list(getprimes(500000))):
		while True:
			sublist = [tup for tup in candidates if tup[1] % prime == 0]
			if len(sublist) > 1:
				subprods = [tup[0]**(tup[1] // prime) for tup in sublist]
		#		print(sublist, subprods)
				sublist.pop(subprods.index(max(subprods)))
				candidates = [tup for tup in candidates if tup not in sublist]
				out.update(sublist)
				print('_____________')
				print(prime, len((candidates)), len(out))
			else:
				break
	answer = [i for i in tuplist if i not in out].pop()
	print(answer)
	return tuplist.index(answer) + 1

	
def prob104():
	old = 0
	fib = 1
	pandigends = []
	for i in range(1,1000000):
		fib = str(fib)
		if len(fib) >8:
			if ispandigital(fib[-9:]):
				pandigends.append(i)
				#if ispandigital(fib[-9:]):
				print(i, fib)
		if len(fib) == len(str(old)) > 30:
			fib = fib[:15]+fib[-15:]
		fib, old = (int(fib) + old, int(fib))
	old = 0
	fib = 1
	for i in range(1,1000000):
		if i in pandigends:
			fib = str(fib)
			if ispandigital(fib[:9]):
				if ispandigital(fib[-9:]):
					print(i,fib)
				else:
					print(f'fails at {i}')
		fib, old = (int(fib) + old, int(fib))
	print(pandigends)
	
	
def prob114():
	@memoized
	def add_block(placed, last_color):
		length = 167
		m = 50
		ways = 0
		if placed == length:
#			print('done')
			return 1
		if placed <= length -m and last_color == 'black':
			#add reds
			for i in range(m, length - placed +1):
#				print('red', i)
				ways += add_block(placed + i, 'red')
		if placed < length:
			# add black
#			print('black')
			ways += add_block(placed +1, 'black')
		return ways
	return add_block(0,'black')
	
def prob122():
	answer = None 
	k = {}
	unfound = set(range(1,201))
	found = set([(0,),(1,)])
	targets = set(range(1,201))
	mults = 0
	def ways(n, vals):
		if n == 1:
			return 1
		numways = {val:0 for val in vals}
		numways[1] = 1
		while True:
			pass
			
	while unfound:
		mults += 1
		newfound = set()
		maxi = max(unfound)
		for tup in found:
			for s in tup:
				for t in tup:
					newval = s + t
					if newval > maxi:
						break
					if newval in unfound:
						k[newval] = mults
						unfound.remove(newval)
					if newval not in tup:
						newfound.add(tuple(sorted(tup + (newval,))))
#					print(val, unfound, found, newfound)
		found = newfound
		print(len(unfound),len(found))
	#	print(mults,found)
	print(k)
	return sum(k.values())
	
	
def prob123():
	target = 10 ** 10
	for m, p in enumerate(getprimes()):
		n= m + 1
		if ((p + 1) ** n  + (p - 1) ** n) % p ** 2 > target:
			print(n,p,)
			return n
			
			
def prob124():
	target = 100000

	rads = {}
	rads[1] = [1]
	viewed = 1
	i = 2
	while True:
		prime_factors = primedivisors(i)
		prime_prod = prod(prime_factors)
		if prime_prod == i:
#			print(i,viewed)
			ns = set([prime_prod])
			#viewed.add(i)
			newns = list([prime_prod, ])
			for prime in prime_factors:
				for n in newns.copy():
					for j in range(1,int(math.log(target//n,prime))+1):
						newns.append(n*prime**j)
#					print(newns)
			ns.update(newns)
			viewed+=len(ns)
			if viewed >= target // 10:
				print(i, sorted(ns))
				return sorted(ns)[-(viewed-(target//10))-2]# dont know why i have to remove 2 here!!!
			rads[i]=ns
		i += 1
		
	primelist = list(getprimes(target))
	factlist = [[1]]
	for prime in getprimes(target):
		for fact in deepcopy(factlist):
			baseval = prod(fact)
			n = int(math.log(target/baseval))
		
	answer = None
	return answer

	
def prob139():
	maxperim = 100000000
	answer = 0
	for a,b,c in geteuclidtriangles(int(math.sqrt(maxperim))):
		p = a + b + c
		if p < maxperim:
			d = math.sqrt(c ** 2 - 2 * a * b)
			if d%1 == 0:
				if c % d == 0:
					answer += maxperim // p
	return answer
	
	
def prob142():
	maxi = 1000000
	squares = [i**2 for i in range(1, int(math.sqrt(maxi)))]
	squareset = set(squares)
	print(len(squares),max(squares))
	for z in range(1,maxi):
		for i,a in enumerate(squares,1):
			if (z+a) > maxi:
				break
			if (2*z+a) in squareset:
	#                       print(z,a,z+a)
				for b in squares[i:]:
					if (z+b) > maxi:
						break
					if (2* z + b) in squareset:
						y = z + a
						x = z + b
#                                               print(x,y,z)
						if all((w in squareset for w in (x+y,x-y,x+z,x-z,y+z,y-z))):
							print(x,y,z)
							return sum([x,y,z])
	return None
	
def prob145():
	num = 0
	for i in range(1,1000000):
		if all((int(x)&0b1 for x in str(i+int(str(i)[::-1])))):
			if i%10!=0:
				num += 1
	return num
	
def prob151():
	@memoized
	def singles(batches, sheets):
		if len(sheets) > batches or batches == 1:
			return 0
		expected = 0
		if len(sheets) == 1:
			expected = 1
		for sheet in sheets:
			newsheets = list(sheets)
			newsheets.remove(sheet)
			if not newsheets:
				newsheets = []
			if int(sheet) < 5:
				newsheets.extend([str(x) for x in range(int(sheet)+1,6)])
			expected += singles(batches-1, ''.join(newsheets))/len(sheets)
	#		print(batches,sheet,sheets,newsheets)
		return expected
	return round(singles(15,'2345'),6)
	
def prob160():
	def trim(n):
		return int(str(n).rstrip('0')[-5:])
	ans = trim(factorial(20))
	adict = {}
	maxi = 0
	for i in range(21,101):
		ans = trim(ans*i)
#               if ans in adict:
#                       print(i, ans, adict[ans], i- adict[ans])
#               adict[ans] = i
	ans1 = ans
	print(ans)
	for i in range(1,2):
		ans = trim(ans*ans1)
	return ans
	
	
def prob164():
	@memoized
	def ways(digit, d1, d2):
		ans = 0
		maxdigit = (9 - d1 - d2)
		if digit == 20:
			return maxdigit + 1
		for i in range(maxdigit+1):
			ans += ways(digit+ 1, d2, i)
		return ans
	answer = 0
	for i in range(1,10):
		answer += ways(2, 0, i)
	return answer
	
def prob181():
	def routes(total, maxval):
		if any((total == 0, total == 1, maxval == 1)):
			return 1
		way = 0
		for i in range(min(maxval, total), 0, -1):
			way += routes(total - i, i)
		return way 
		
	def w_ways(w,max_w):
#		if w == 1:
#			return 1
		if w== 0:
			return 1
		way = 0
		for j in range(max_w,-1,-1):
			way += w_ways(w-j, min(w-j,j))
		return way
		
	def ways(b,w,max_b,max_w,chain = []):
		if b == 0:
			way = 0
			for i in range(w+1):
				way += routes(w, max_w)# correct this
			return way
		way = 0
		for i in range(max_b, 0, -1):
			for j in range(max_w, -1, -1):
				ch = 'b' * b + 'w' * w
				way += ways(b-i,w-j,min(b-i,i),min(w-j,j),chain.append(ch))
				print(i,j,way)
		return way

	b = 3
	w = 1 #1 : 7,  2: 15?
	return ways(b,w,b,w)
	
def prob187():
	target = 10**8
	answer = 0
	for i in getprimes(target//2):
		for j in getprimes(min(target//i,i)):
			answer += 1

	return answer
	
	
def prob205():
	def throw(n,l=[1.]):
		m = [0. for i in range(len(l)+n)]
		for i, p in enumerate(l):
			for j in range(1,1+n):
				m[i+j] += p * 1/n
		return m
	peter = [1]
	colin = [1]
	for i in range(9):
		peter = throw(4, peter)
	for i in range(6):
		colin = throw(6, colin)
	return round(sum([c*sum(peter[i:]) for i, c in enumerate(colin,1)]),7)
	
def prob206():
	l=[]
	for i in range(100000 , 1000000):
		l.append( int(''.join(k+m for k,m in zip('12345678',str(i)+'55'))+'900'))
	print('listed')
	l.sort(key=lambda x: (math.sqrt(x))%1)
	print('sorted')
	for i in range(-100,100):
		for j in range(10,100):
			s = int(str(l[i])[:12]+''.join(k + l for k,l in zip('78',str(j)))+'900')
			if int(math.sqrt(s)) ** 2 == s:
				return math.sqrt(s)
				
def prob214():
	target = 100
	answer = 0
	factors = {}
	for i in getprimes(target//2):
		for j in getprimes(min(target // i, i)):
			totient = (i - 1) * (j - 1)
			

	return answer
				
def prob230():
	answer = 0
	s = {}
	s[1] = '1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679' #'1415926535'
	s[2] = '8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196' # '8979323846'
	lens = {}
	lens[1] = len(s[1])
	lens[2] = len(s[2])
	
	for i in range(18):
		n = (7**i)*(127+(19*i))
		a = lens[1]
		b = lens[2]
		index = 2
		while n>b:
			a,b = b, a+b
			index+=1
			lens[index] = b
#			print(index, lens[index])
		while index>2:
			if n>lens[index-2]:
				n -= lens[index-2]
				index -= 1
			else:
				index -=2
				
#			print(index, n)
		if index == 1 and n > len(s[1]):
			answer += 10**i*int(s[2][n-101])
		else:
			answer += 10**i*int(s[index][n-1])
	return answer
				
def prob243():
	target = 15499/94744
	print (f"target = {target}")
	lowest = 1
	base = prod([2,3,5,7,11,13,17,19,23])
	for d in range(base,1300000*base, base):
		divs = primedivisors(d)
		resiliants = 0 #set()
		for n in range(3,d,2): #  check only odd numbers
			for div in divs: #
				if (d % div == 0) and (n % div == 0):
					resiliants += 1
					break
		resiliance = 1-((resiliants + (d/2 - 1)) / (d -1))
		if resiliance < lowest:
			lowest = resiliance
			print(d, resiliance, resiliance < target, target)
			
def prob243a():
	divlist = []
	reslist = [7147140,7657650,8168160,8678670,9189180,9699690,19399380]
	divlist = sorted([d for d in divisors(reslist[0], set())])
	for i in reslist[1:]:
		divs = set()
		for div in divisors(i, divs):
			pass
			
			
def prob277():
	target = 'UDDDUdddDDUDDddDdDddDDUDDdUUDd' #'DdDddUUdDDDdUDUUUdDdUUDDDUdDD'# 'DdDddUUdDD'
	answers = []
	def f0(n): return n // 3, 'D'
	
	def f1(n): return  ((4*n)+2) // 3, 'U'
	
	def f2(n): return  ((2*n)-1) // 3, 'd'
	
	def fd(n): return ((3*n)+1) // 2
	
	def fD(n): return 3 * n
	
	def fU(n): return ((3*n)-2) // 4
	
	f = {0:f0, 1:f1, 2:f2}
	
	decode = {'d':fd, 'D':fD, 'U':fU}
	for i in range(1,100000000):
		a = i
		for l in reversed(target):
			a0 = a
			a = decode[l](a)
#			print(i,a,a0)
			if a0 != f[a%3](a)[0]:
				break
		else:
			print(l,i,a)
			if a>10**15:
				return a
		
	
	for i in range(1,400000000):
		a = i*337497
		s = ''
		while a>1:
			a,l = f[a%3](a)
			s+= l

		if s[:len(target)] == target:
			print(i, s)
			answers.append(i)
	ans0 = 0
	
	for ans in answers:
		print(ans,ans0, ans-ans0)
		ans0 = ans
		
	answer = None 
	return answer
	
def prob284():
	answer = 0
	b = 14 # base 14
	steadies = [(0,0),]
	for i in range(1000):
		newsteadies = []
		digitval = b**i
		seuil = b ** (i+1)
		for ds,val in steadies:
			for d in range(b):
				n = val + (digitval*d)
				if pow(n,2, seuil) == n:
					dsum = d+ds
					newsteadies.append((dsum,n))
					if d != 0:
						answer += dsum
		steadies = newsteadies
	print(answer)
	return n_in_base(answer, b)
	
			
def prob293():
	admissibles = [1]
	target = 10**9
	pgen = getprimes(23)
	full_adds = []
	for p in pgen:
		new_ads = []
		for a in admissibles:
			if a <= target / p:
				new_ads.extend([a * (p ** i) for i in range(1, int(math.log(target/a, p))+1)])
			else:
				break
	#	print(p,new_ads)
		full_adds.extend(new_ads)
		new_ads.sort()
		admissibles = new_ads
	full_adds.sort()
	print(len(full_adds))
	pgen = getprimes(target+1000)
	p = 2
	pseudos = set()
	for a in full_adds:
		if a +2> p:
			p = a + 3
		while not isprime(p):
			p += 2
#		print(a, m)
		pseudos.add(p-a)
	return sum(pseudos)
	
	
def prob297():
	fibgen = getfibonacci()
	target = 10**6
	next(fibgen)
	fibs = {1:next(fibgen),
					2:next(fibgen),
					3:next(fibgen),
					}
	zeck_sum = {1:1, 2:2, 3:3, 4:5}
	
	fib1 = 2
	fib2 = 3
	for fib in fibgen:
		fib1, fib2 = fib2, fib
		num = fib1 + fib2 -1
		zeck_sum[num] = fib1 + zeck_sum[fib1-1] + zeck_sum[fib2-1]
#		print(num, sum(zeck_sum.values())+1)
		if fib > target:
			num = fib1 -1
			break
	sums = sorted(zeck_sum.keys())
	sums = [n for n in sums if n < target]
	
	remainder = target - max(sums)
	somme = zeck_sum[num]
	while remainder:
		sums = [n for n in sums if n < target]
		break
		
	
	print(zeck_sum, num)
	answer = None 
	return answer
			
			
def prob301():
	@memoized
	def addone(n):
		if n == 29:
			return 2
		# add a zero
		summ = addone(n+1)
		# add a one
		if n < 28:
			summ += addone(n + 2)
		else:
			summ += 1
		return summ
	return addone(0)
	
def prob315():
	segments = {'0' : [1, 1, 1, 0, 1, 1, 1] ,
							'1' : [0, 0, 1, 0, 0, 1, 0] ,
							'2' : [1, 0, 1, 1, 1, 0, 1] ,
							'3' : [1, 0, 1, 1, 0, 1, 1] ,
							'4' : [0, 1, 1, 1, 0, 1, 0] ,
							'5' : [1, 1, 0, 1, 0, 1, 1] ,
							'6' : [1, 1, 0, 1, 1, 1, 1] ,
							'7' : [1, 1, 1, 0, 0, 1, 0] ,
							'8' : [1, 1, 1, 1, 1, 1, 1] ,
							'9' : [1, 1, 1, 1, 0, 1, 1]
							}
	saving = {}
	for d in string.digits:
		for e in string.digits:
			saving[(d,e)] = 2* sum([a and b for a,b in zip(segments[d], segments[e])])
#	print(len(saving))
	for i in [x for x in range(100) if x%10 in [1,3,7,9]]:
		for j in range(digitsum(i)+1,65):
			saved = 0
			old = str(i).zfill(2)
			summ = j
			while summ > 9:
				for k, n in enumerate(str(summ)[::-1]):
					saved += 2* sum([a and b for a,b in zip(segments[n], segments[old[-k-1]])])
				old, summ = str(summ), digitsum(summ)
			for k, n in enumerate(str(summ)[::-1]):
				saved += 2* sum([a and b for a,b in zip(segments[n], segments[old[-k-1]])])
			saving[(i,j)] = saved
#	print(len(saving))
	answer = 0
	for prime in getprimes(2*10**7):
		if prime > 10**7:
			answer += saving[(prime % 100, digitsum(prime))]
	return answer
			
def prob345():
	def matrixsum(l):
		return sum(m[r][c] for c, r in enumerate(l))
	m=[[int(y) for y in x.split(' ')] for x in '''7 53 183 439 863 497 383 563 79 973 287 63 343 169 583
	627 343 773 959 943 767 473 103 699 303 957 703 583 639 913
	447 283 463 29 23 487 463 993 119 883 327 493 423 159 743
	217 623 3 399 853 407 103 983 89 463 290 516 212 462 350
	960 376 682 962 300 780 486 502 912 800 250 346 172 812 350
	870 456 192 162 593 473 915 45 989 873 823 965 425 329 803
	973 965 905 919 133 673 665 235 509 613 673 815 165 992 326
	322 148 972 962 286 255 941 541 265 323 925 281 601 95 973
	445 721 11 525 473 65 511 164 138 672 18 428 154 448 848
	414 456 310 312 798 104 566 520 302 248 694 976 430 392 198
	184 829 373 181 631 101 969 613 840 740 778 458 284 760 390
	821 461 843 513 17 901 711 993 293 157 274 94 192 156 574
	34 124 4 878 450 476 712 914 838 669 875 299 823 329 699
	815 559 813 459 522 788 168 586 966 232 308 833 251 631 107
	813 883 451 509 615 77 281 613 459 205 380 274 302 35 805'''.splitlines()]
	maxbycols = [i for i in range(len(m))]
	shuffle(maxbycols)
	while True:
		summ = matrixsum(maxbycols)
		for i in range(len(m)-1):
			try:
				for j in range(i+1,len(m)):
					c, r = i, maxbycols[i]
					c1, r1 = j, maxbycols[j]
					if m[r][c1] + m[r1][c] > m[r][c] + m[r1][c1]:
						maxbycols[i], maxbycols[j] = maxbycols[j] , maxbycols[i]
			except IndexError:
				pass
		if summ == matrixsum(maxbycols):
			break
		else:
			print(matrixsum(maxbycols), maxbycols)
	return matrixsum(maxbycols)
	
def prob347():
	target= 10**7
	answers = []
	js = []
	for i in getprimes(target // 2):
		for j in getprimes(min(i, target // i)):
			if j >= i:
				continue
			maxas=[]
			for a in range(int(math.log(target/(i*j),i)+1)):
				b = int(math.log(target/(i*j*i**a),j))
				maxas.append(i*j*i**a*j**b)
	#			print(i,j,a,b,maxas[-1])
#			print('max',i,j,max(maxas))
			answers.append(max(maxas))
	return sum(answers)
	
def prob348():
	squares = set(getsquares(1000000000))
	mids = ['']+list(string.digits)
	answers = []
	for i in range(1,1*10** 4):
		start = str(i)
		stop = str(i)[::-1]
		for mid in mids:
			pal = int(start+mid+stop)#f'{i}{mid}{stop}')
			hits = 0
			for cube in getcubes(pal):
				if pal- cube in squares:
					hits += 1
			if hits == 4:
				answers.append(pal)
		if len(answers) == 5:
			break
	answers.sort()
	return sum(answers[:5])


def prob357():
	target = 10**5
	primeset = set(getprimes(target+1))
	print('set')
	
	def finish_ints(basefacts,ps):
		nonlocal target, primeset
		summ = 0
		for i, p in enumerate(ps):
			facts = basefacts + [p]
			val = prod(facts)
#			print(val,facts)
			if val <= target:
				if (val+1) in primeset:
					if all(((d+val//d) in primeset for d in facts)):
						summ += val
	#					print ('yes',val,facts)
			else:
				break
		return summ

	def build_ints(basefacts,ps):
		nonlocal target, primeset
		summ = 0
		for i, p in enumerate(ps):
			facts = basefacts + [p]
			val = prod(facts)
#			print(val,facts)
			if val <= target:
				if (val+1) in primeset:
					if all(((d+val//d) in primeset for d in facts)):
						summ += val
	#					print ('yes',val,facts)
				if p< target - val:
					summ += build_ints(facts,ps[i+1:])
				else:
					summ += finish_ints(facts[:-1],ps[i+1:])
					break
			else:
				break
		return summ
				
	answer = 3
	ps = list(getprimes(target//2))[1:]
	print('listed')
	#ps.reverse()
	return build_ints([2],ps)
	
def prob357(): 
	target = 10**3
	answer = 3
	primeset = set(getprimes((target // 2) + 2))
	for p in getprimes(target):
		val = p -1
		if val % 4 != 0:
			if val % 9 != 0:
				for div in primedivisors(val):
					if (val // div) + div not in primeset:
						break
				else:
		#			print(val)
					answer += val
	return answer
	
	
def prob357(): 
	target = 10**6
	answer = 1 # 1 
	primes = getprimes((target // 2) + 2)
	primeset = set(primes)
	primes = getprimes((target // 2) + 2)
	facts = {2:[2]}
	next(primes)
	print('set',len(primeset))
	#build factors of composite numbers with 2 + at least 1 other prime, no prime more than once
	products = [2]
	bigproducts = []
	for p in primes:
		newproducts = []
		for i, product in enumerate(products):
			if product < target // p:
				newprod = product * p
				newproducts.append(newprod)
				facts[newprod] = facts[product]+[p]
			else:
				bigproducts.extend(products[i:])
				#print(products, i)
				del products[i:]
				break
		products.extend(newproducts)
		products.sort()
	print(len(products), len(bigproducts), len(facts.keys()))
	for prod,divs in facts.items():
		if all(d+prod//d in primeset for d in divs):
			if len(divs)>3:
				print(divs)
			answer += prod
	return answer
	
def prob387():
	harshads = list(range(1,9))
	answer = 0
	odds = set([1,3,7,9])
	for i in range(13):
		harsh = list()
		for h in harshads:
			strong = isstrongharshad(h)
			for j in range(10):
				n = 10 * h + j
				if strong:
					if isprime(n):
						answer += n
				if isharshad(n):
					harsh.append(n)
		harshads = list(harsh)
#		print(i, (harsh))
	return answer
	
	
def prob407():
	target = 10**3
	summ = 1
	ps = getprimes(target)
	monoprimes = set()
	
	for p in ps:
		i = 1
		q = p
		while q <= target:
			monoprimes.add(q)
			i += 1
			q = p ** i
			
	ivals = iter(range(2, target+1))
	for i in ivals:
		if i in monoprimes:
			summ += 1
			continue
		for a in range(i,0,-1):
			if pow(a,2,i) == a:
				summ += a
	#			print(i,a)
				break
	return summ
	
def prob424():
	url = 'https://projecteuler.net/project/resources/p424_kakuro200.txt'
	r = requests.get(url)
	puzzles = r.text.splitlines()
	for puzzle in puzzles:
		puzzle_text = puzzle
		puzzle = ''
		while '(' in puzzle_text:
			s = puzzle_text.split('(',1)
			start, next = s[0], s[1]
			s = next.split(')',1)
			mid, end = s[0].replace(',',';'), s[1]
			puzzle += start
			puzzle += mid
			if '(' not in end:
				puzzle += end
			puzzle_text = end

		unused_nums = set(range(10))
		unused_lets = set('ABCDEFGHIJ') 
		mappings = {l : set(string.digits) for l in unused_lets}
		size = int(puzzle[0])
		assert 5 < size  < 8
		grid = [list()for i in range(size)]
		nums = {}
		
		for i,square in enumerate(puzzle.split(',')[1:]):
			grid[i//size].append(square)
		for i, row in enumerate(grid):
			for j, cell in enumerate(row):
				if cell in 'ABCDEFGHIJ':
					nums[(i,j,'X')] = {'sum':cell, 'nums':[(i,j)]}
				if cell[0] in 'hv':
					head = cell.split(';')
					for h in head:
						num = []
						num_sum = h[1:]
						if h[0] == 'h':
							for n in range(j+1,size):
								nextcell = grid[i][n]
								if nextcell in 'ABCDEFGHIJO':
									num.append((i,n))
								else:
									break
							nums[(i,j,'h')] = {'sum':num_sum, 'nums':num}
						if h[0] == 'v':
							for n in range(i+1,size):
								nextcell = grid[n][j]
								if nextcell in 'ABCDEFGHIJO':
									num.append((n,j))
								else:
									break
							nums[(i,j,'v')] = {'sum':num_sum, 'nums':num}
		for key, val in sorted(nums.items(), key = lambda  x: x[1]['nums']):
			if len(val['nums']) == 2:
				if len(val['sum']) == 2:
					one = val['sum'][0]
					mappings[one] = {1}
					unused_nums.discard(1)

			if len(val['nums']) == 3:
				if len(val['sum']) == 2:
					onetwo = val['sum'][0]
					mappings[onetwo] = {1,2}.union(unused_nums)
					if len(mappings[onetwo]) == 1:
						unused_nums.discard(2)

		print(mappings)
				
#		print( nums)
					
		assert len(grid[-1]) == size
		print('grid\n','\n'.join([str(line) for line in grid]))
			

	answer = None 
	return answer
				
	
def prob493():
	probs = {}
	
	def tire(balls, newcol, colours, round, prob):
	#       if prob < 1.e-11:
#                       return
		nonlocal probs
		if newcol:
			balls += 9
			colours += 1
		else:
			balls -= 1
		if round == 19:
			probs[colours] += prob
			return
		if balls:
			tire(balls, False, colours, round + 1, prob*(balls/(69-round)))
		if colours < 7:
			tire(balls, True, colours, round + 1, prob*(70-(colours*10))/(69-round))
			
	for i in range(1,8):
		probs[i] = 0.
	colours = 0
	balls = 0
	tire(balls,True,colours,0,1)
	
	return round(sum(i*j for i,j in probs.items()),9)
	
def prob621():
	answer = 0
	target = 100000000
	tris = [0]
	tris.extend(list(gettrianglenum(target)))
	tri3s = set(tris)
	for tri1 in tris:
		for tri2 in tris:
			tri3 = target - tri1 - tri2
#			print(tri1,tri2,tri3)
			if tri3 < 0:
				break
			if tri3 in tri3s:
				answer += 1
#				print('yes')
				
	return answer
	
	
def prob630():
	answer = 0
	
	points = []
	s = 290797
	for i in range(2500):
		s = s ** 2 % 50515093
		x = s % 2000 - 1000
		
		s = s ** 2 % 50515093
		y = s % 2000 - 1000
		points.append((x,y))
		
	lines = dict()
	for i, pt1 in enumerate(points):
		for pt2 in points[i+1:]:
			try:
				a = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
				b = pt1[1] - a * pt1[0]
				a = round(a,8)
			except ZeroDivisionError:
				a = 'inf'
				b = pt1[0]
			if a in lines:
				lines[a].add(round(b,8))
			else:
				lines[a] = set((round(b,8),))
	num_lines = sum([len(x) for x in lines.values()])
	print(num_lines, 'lines')

	for line in lines.values():
		answer += (num_lines - len(line))*len(line)
	return answer
	
	
if __name__ == "__main__":
	prob = 'smath'
	if type(prob) == str:
		solvemaths1()
	elif f'prob{prob}' in locals():
		start = time.clock()
		answer = eval(f'prob{prob}()')
		time_taken = time.clock() - start
		print(f'safari-https://projecteuler.net/problem={prob}')
		print(f'answer = {answer} found in {round(time_taken,3)} seconds, copied to the clipboard')
		clipboard.set(str(answer))
		
	else:
		print(f'def prob{prob}(): \n\tanswer = None \n\treturn answer')
		clipboard.set(f'def prob{prob}():\n\tanswer = None \n\treturn answer')
	p = requests.get('https://projecteuler.net/profile/dancergraham.png')
	img = Image.open(io.BytesIO(p.content))
	img.show()
	

