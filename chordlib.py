from collections import namedtuple, defaultdict
import itertools
from math import exp, pi, sin, cos
from fractions import Fraction
import pickle
import sympy
from sympy import Poly
import copy

Chord = namedtuple("Chord", ["n", "endpoints"])
import numpy as np
from pyx import *


class Basis:
	def __init__(self, basis):
		self.basis = basis
		self.n = len(basis)
		self.dct = dict()
		for i, basis in enumerate(self.basis):
			self.dct[basis] = i
	
	def vector_index(self, vector):
		return self.dct[vector]
	
	def to_coordinates(self, LC):
		coords = [0] * self.n
		for coeff, gen in to_LC(LC).lc:
			coords[self.dct[gen]] += coeff
		return coords
	
	def to_linear_combination(self, coords):
		def to_int_if_possible(x):
			if float(x).is_integer():
				return int(x)
			return x
		return LinearCombinationOfCD([ Term(to_int_if_possible(c), self.basis[i]) for i, c in enumerate(coords) if c != 0 ])
	
	def to_matrix(self, vectors, sortandgroup=True, integer=True):
		if len(vectors) == 0:
			return np.matrix(np.zeros(self.n))
		coords = [self.to_coordinates(c) for c in vectors]
		if sortandgroup:
			new_coords = []
			coords.sort()
			for vect, group in itertools.groupby(coords):
				new_coords.append(vect)
			coords = new_coords
		if integer:
			return np.matrix(coords, np.dtype(object))
		return np.matrix(coords)

	def permute(self, lst):
		#check if lst is a permutation
		slst = sorted(lst)
		assert slst == list(range(len(lst)))
		return Basis( [self.basis[i] for i in lst] )

class Relations:
	def __init__(self, basis, relations):
		self.basis = basis
		self.relations = relations
		rels, total = relations.shape
		if not relations.any():
			rels = 0
		self.rels = rels
		self.indep = total - rels
		self.total = total
	def apply_relations(self, lc):
		lc = to_LC(lc)
		if self.rels == 0:
			return lc
		new_terms = []
		for coeff, gen in lc.lc:
			vi = self.basis.vector_index(gen)
			if vi < self.rels:
				relcoeff =  float(self.relations[vi, vi])
				if relcoeff.is_integer():
					relcoeff = int(relcoeff)
				relcoeff = sympy.sympify(relcoeff)
				assert relcoeff != 0
				new_lc = -sympy.sympify(coeff/relcoeff)*(self.basis.to_linear_combination(self.relations[vi].A1) - relcoeff*to_LC(Term(1, gen)) )
				new_terms += new_lc.lc
			else:
				new_terms.append(Term(coeff, gen))
		return LinearCombinationOfCD(new_terms, simplify=True)
		
class CDSimplifier:
	def __init__(self, ESoptimization=True, SLoptimization=True):
		self.relations = defaultdict(lambda : None)
		self.ESoptimization=ESoptimization
		self.SLoptimization=SLoptimization
		
	def add_relations(self, relations, strands, chords):
		self.relations[(strands, chords)] = relations
	
	def simplify_term(self, term):
		coeff, gen = term
		c = gen.number_of_chords
		if not (gen.n, c) in self.relations.keys():
			print("Could not simplify a term with %i strands and %i chords" % (gen.n, c))
			return to_LC(term)
		return self.relations[(gen.n, c)].apply_relations(term)
	
	def simplify_term_ES_SL(self, term):
		coeff, gen = term
		return self.simplify_term_ES(Term(coeff, gen.move_small_loops()))
		
	def simplify_term_SL(self, term):
		coeff, gen = term
		return self.simplify_term(Term(coeff, gen.move_small_loops()))
	
	def simplify_term_ES(self, term):
		"""apply th Empty Strand optimization"""
		coeff, gen = term
		if 0 in gen.signature:
			if len(gen.endpoints)==1:
				return to_LC(term)
			zeros = [0 if i == 0 else 1 for i in gen.signature ]
			if len(zeros) == gen.n:
				return to_LC(term)
			oc = gen._to_old_chord()
			for i, z in list(enumerate(zeros))[::-1]:
				if z == 0:
					oc.endpoints.pop(i)
			oc = Chord(len(oc.endpoints), oc.endpoints)
			simplified = self.simplify_term(Term(coeff, ChordDiagram.from_old_chord(oc)))
			new_simplified = []
			for coeff, gen in simplified.lc:
				oc = gen._to_old_chord()
				for i, z in enumerate(zeros):
					if z == 0:
						oc.endpoints.insert(i, [])
				oc = Chord(gen.n, oc.endpoints)
				#~ print(gen.n. oc.endpoints)
				new_simplified.append(Term(coeff, ChordDiagram.from_old_chord(oc)))
			return to_LC(new_simplified)
		else:
			return self.simplify_term(term)
		
	def simplify(self, lc, ESoptimization=True, SLoptimization=True):
		lc = to_LC(lc)
		if self.ESoptimization:
			if self.SLoptimization:
				return sum( [self.simplify_term_ES_SL(t) for t in lc.lc], to_LC([])).simplify()
			else:
				return sum( [self.simplify_term_ES(t) for t in lc.lc], to_LC([])).simplify()
		else:
			if self.SLoptimization:
				return sum( [self.simplify_term_SL(t) for t in lc.lc], to_LC([])).simplify()
			else:
				return sum( [self.simplify_term(t) for t in lc.lc], to_LC([])).simplify()

	def save_relation(self, filename, n, c):
		with open(filename, 'wb') as f:
			pickle.dump(self.relations[(n, c)], f)
	def load_relation(self, filename, n, c):
		with open(filename, 'rb') as f:
			data = pickle.load(f)
		self.add_relations(data, n, c)
	
	def save(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(dict(self.relations), f)
	@classmethod
	def load(cls, filename):
		r = cls()
		with open(filename, 'rb') as f:
			data = pickle.load(f)
		for s, c in data.keys():
			r.add_relations(data[(s, c)], s, c)
		return r
		
class ChordDiagram:
	def __init__(self, n, endpoints):
		""" The format is as follows: the endpoints is a list of lists.
		Each of this lists contains a series of tuples (strand, endpoint),
		which tells you where is the point connected to."""
		self.n = n
		self.endpoints = endpoints
		
	def is_valid(self):
		endpoints = copy.deepcopy(self.endpoints)
		if len(self.endpoints) != self.n:
			return (False, 'incorrect number strands')
		for i in range(self.n):
			for j in range(self.signature[i]):
				if isinstance(endpoints[i][j], tuple):
					k, l = endpoints[i][j]
					if isinstance(endpoints[k][l], tuple):
						if not endpoints[k][l] == (i, j):
							return (False, 'the endpoint (%i, %i) does not point back to (%i, %i)' % (k, l, i, j))
						else:
							endpoints[i][j] = -1
							endpoints[k][l] = -1
						
					else:
						return (False, 'the endpoint (%i, %i) is %r, not an endpoint' % (k, l, str(endpoints[k][l])))
		for s in endpoints:
			for e in s:
				if e != -1:
					return (False, 'encountered %r instead of -1 after removing all chords' % str(e))
		return (True, "all is fine")
	
	def assert_valid(self):
		value, message = self.is_valid()
		if not value:
			raise ValueError(' %r is not valid, the reason is %r' % (str(self), message))
			
	
	@property
	def number_of_chords(self):
		return sum(self.signature)//2
	
	@property
	def signature(self):
		return [len(s) for s in self.endpoints]
	
	def __hash__(self):
		return hash(str(self.endpoints))
		
	def __eq__(self, other):
		return self.endpoints == other.endpoints

	def __lt__(self, other):
		if self.n != other.n:
			return self.n < other.n
		if self.number_of_chords != other.number_of_chords:
			return self.number_of_chords < other.number_of_chords
		if self.signature != other.signature:
			return self.signature < other.signature
		return self.endpoints < other.endpoints

	def __gt__(self, other):
		if self.n != other.n:
			return self.n > other.n
		if self.number_of_chords != other.number_of_chords:
			return self.number_of_chords > other.number_of_chords
		if self.signature != other.signature:
			return self.signature > other.signature
		return self.endpoints > other.endpoints

	def __str__(self):
		return "Chord with "+str(self.n)+" strands: "+str(self.endpoints)

	def __mul__(self, other):
		"""self is under other"""
		if self.n != other.n:
			raise ValueError("different numbers of strands")
		new_endpoints = copy.deepcopy(other.endpoints)
		for i in range(self.n):
			new_endpoints[i] += [ (strand, endpoint+len(other.endpoints[strand])) for strand, endpoint in self.endpoints[i] ]
		res = self.__class__(self.n, new_endpoints)
		res.assert_valid()
		return res
	#~ def __mul__(self, other):
		#~ """self is over other"""
		#~ if self.n != other.n:
			#~ raise ValueError("different numbers of strands")
		#~ print('Multiplying %r and %r' % (str(self), str(other)))
		#~ new_endpoints = copy.deepcopy(self.endpoints)
		#~ for i in range(self.n):
			#~ new_endpoints[i] += [ (strand, endpoint+len(self.endpoints[strand])) for strand, endpoint in other.endpoints[i] ]
		#~ res = self.__class__(self.n, new_endpoints)
		#~ res.assert_valid()
		#~ return res
		
	def _to_old_chord(self):
		endpoints = copy.deepcopy(self.endpoints)
		chord = 0
		for strand in endpoints:
			for i, endpoint in enumerate(strand):
				if isinstance(endpoint, tuple):
					otherstrand, no =  endpoint
					strand[i] = chord
					endpoints[otherstrand][no] = chord
					chord += 1
		return Chord(self.n, endpoints)

	def to_pairs(self):
		"""(j, i) , (j', i') means a chord connects j-th position on strand i with j'-th position on strand i' """
		endpoints = copy.deepcopy(self.endpoints)
		pairs =[]
		for strand in endpoints:
			for i,d in enumerate(strand):
				if isinstance(d, tuple):
					sother, eother = d
					strand[i] = -1
					dp = endpoints[sother][eother]
					endpoints[sother][eother] = -1
					pairs.append( (dp, d) )
		return pairs
	
	def S(self, which):
		def new_info(s, e):
			if s == which:
				return (s, self.signature[which] - e - 1)
			return (s, e)
		endpoints = copy.deepcopy(self.endpoints)
		for s in range(self.n):
			for j in range (self.signature[s]):
				endpoints[s][j] = new_info(*self.endpoints[s][j])
		endpoints[which] = endpoints[which][::-1]
		res =  ChordDiagram(self.n, endpoints)
		res.assert_valid()
		return res
	
	def has_small_loop(self):
		for s in range(self.n):
			for i in range(self.signature[s]-1):
				if self.endpoints[s][i] == (s, i+1):
					return True
		return False

	def move_small_loops(self):
		#~ print("Moving small loops on %s" % str(self))
		def move_small_loops_strand(strand):
			scopy = copy.deepcopy(strand)
			#~ print (scopy)
			i = 0
			SL = []
			while 1:
				if i+1 < len(scopy):
					if scopy[i] == scopy[i+1]:
						scopy.pop(i+1)
						SL.append(scopy.pop(i))
						i = max(0, i-1)
					else:
						i += 1
				else:
					break
			for i in SL:
				scopy = [i, i] + scopy
			return scopy
		if self.has_small_loop():
			oc = self._to_old_chord()
			noc = Chord(self.n, [ move_small_loops_strand(s) for s in oc.endpoints ])
			return ChordDiagram.from_old_chord(noc)
		else:
			return self
	
	@classmethod
	def from_old_chord(cls, oldchord):
		n, endpoints = oldchord
		def to_chords(elist):
			"""(e, j, i) , (e, j', i') means chord e connects j-th position on strand i with j'-th position on strand i' """
			new_endpoints = []
			for i in range(n):
				new_endpoints += [(e, j, i) for j,e in enumerate(elist[i])]
			new_endpoints.sort(key=lambda x: x[0])
			pairs = zip(new_endpoints[::2], new_endpoints[1::2])
			return pairs
		return cls.from_chords_list(oldchord.n, to_chords(endpoints))
		
	@classmethod
	def from_chords_list(cls, n, clist):
		clist = list(clist)
		signature = [0]*n
		for c1, c2 in clist:
			e, j, i = c1
			signature[i] = max(signature[i], j+1)
			e, j, i = c2
			signature[i] = max(signature[i], j+1)
		endpoints = [[-1]*signature[i] for i in range(n)]
		for c1, c2 in clist:
			e, j, i = c1
			ee, jj, ii = c2
			endpoints[i][j] = (ii, jj)
			endpoints[ii][jj] = (i, j)
		res = cls(n, endpoints)
		res.assert_valid()
		return res
		
	@classmethod
	def word_to_chord(cls, word):
		res = cls(n=3, endpoints = [[], [], []])
		X =  cls(n=3, endpoints = [[(1, 0)], [(0, 0)], []])
		Y =  cls(n=3, endpoints = [[], [(2, 0)], [(1, 0)]])
		for c in word:
			if c == 'x':
				res = res*X 
			elif c == 'y':
				res = res*Y
		res.assert_valid()
		return res

	def fourterm(self, strand, after, SLoptimization=True):
		terms = []
		terms.append(Term(1, copy.deepcopy(self)))

		old_chord = self._to_old_chord()
		if old_chord.endpoints[strand][after] == old_chord.endpoints[strand][after+1]:
			raise ValueError("Cannot write 4T relation for %r for strand %i after %i" % (str(self), strand, after))
		old_chord.endpoints[strand][after], old_chord.endpoints[strand][after+1] = old_chord.endpoints[strand][after+1], old_chord.endpoints[strand][after] 
		terms.append(Term(-1, ChordDiagram.from_old_chord(old_chord)))
		
		new_strand, new_pos = self.endpoints[strand][after+1]
		old_chord = self._to_old_chord()
		old_chord.endpoints[new_strand].insert(new_pos+1, old_chord.endpoints[strand][after])
		if new_strand == strand and new_pos < after+1:
			old_chord.endpoints[strand].pop(after+1)
		else:
			old_chord.endpoints[strand].pop(after)
		terms.append(Term(-1, ChordDiagram.from_old_chord(old_chord)))
		
		new_strand, new_pos = self.endpoints[strand][after+1]
		old_chord = self._to_old_chord()
		old_chord.endpoints[new_strand].insert(new_pos, old_chord.endpoints[strand][after])
		if new_strand == strand and new_pos < after:
			old_chord.endpoints[strand].pop(after+1)
		else:
			old_chord.endpoints[strand].pop(after)	
		terms.append(Term(1, ChordDiagram.from_old_chord(old_chord)))
		[gen.assert_valid() for c, gen in terms]
		if SLoptimization:
			terms = [ Term(coeff, gen.move_small_loops()) for coeff, gen in terms ]
		return terms

	def all4T(self, SLoptimization = True):
		res = []
		for i in range(self.n):
			for j in range(self.signature[i]-1):
				if self.endpoints[i][j] != (i, j+1):
					res.append(self.fourterm(i, j, SLoptimization))
		return res

	@classmethod
	def all_diagrams(cls, strands, chords):
		def decompositions(n, k):
			""" All possible ways to write n as an ordered sum of k nonnegative numbers"""
			if k == 1:
				return [ [n] ]
			if n == 0:
				return [ [0]*k ]  
			res = []
			for i in range(n+1):
				subdecompositions = decompositions(n-i, k-1)
				res += [ [i] + subdec for subdec in subdecompositions]
			return res
		
		def pairings(lst):
			""" All possible pairings (not ordered) of the elements of the list """
			if len(lst) == 2:
				return [ [tuple(lst)] ]
			if len(lst) == 0:
				return [ [] ]
			first = lst[0]
			res = []
			for i, second in enumerate(lst[1:]):
				res += [[(first, second)] + pairing for pairing in pairings(lst[1:i+1] + lst[i+2:])]
			return res
		decs = decompositions(2*chords, strands)
		pairs = pairings( list(range(2*chords)) )
		res = []
		for dec, pair in itertools.product(decs, pairs):
			d = [None]*(2*chords)
			for e1, e2 in pair:
				d[e1] = e2
				d[e2] = e1
			endpoint = 0
			endpoints = []
			endpoints_to_strands = sum([[i]*strandlen for i, strandlen in enumerate(dec)], [])
			endpoints_to_strand_endpoints = [None]*(2*chords)
			strand = 0
			used_so_far = 0
			for i in range(2*chords):
				endpoints_to_strand_endpoints[i] = i - sum(dec[:endpoints_to_strands[i]])
			for strand, length in enumerate(dec):
				endpoints.append([])
				for i in range(length):
					endpoints[strand].append( (endpoints_to_strands[d[endpoint]], endpoints_to_strand_endpoints[d[endpoint]] ) )
					endpoint += 1
			res.append( cls(strands, endpoints) )
			res[-1].assert_valid()
		return res
	
	@staticmethod
	def tensor(CD1, CD2):
		res = ChordDiagram(CD1.n+CD2.n, copy.deepcopy(CD1.endpoints)+copy.deepcopy(CD2.endpoints))
		nochords = CD1.number_of_chords
		for strand in range(CD1.n, CD1.n+CD2.n):
			res.endpoints[strand] = [(CD1.n+s, endpoint) for s, endpoint in res.endpoints[strand]]
		return res

Term =  namedtuple("Term", ("coeff", "generator"))
def coeff_latex(coeff, first = False):
	if coeff == 1:
		if first:
			txt = ''
		else:
			txt = '+'
	elif coeff == -1:
			txt = '-'
	else:
		txt = sympy.latex(coeff)
		if not first:
			if (txt[0] not in '+-' )or len(txt)==0:
				txt = '+' + txt
	return '$' + txt + '$'
	
def iscoefficient(obj):
	try: 
		sympy.latex(obj)
	except:
		return False
	else:
		return True
		
def function_to_coeffs(function, var, order):
	return Poly(sympy.series(function, var, n=order+1).removeO()).all_coeffs()[::-1]

class LinearCombinationOfCD:
	def __init__(self, lc, simplify = False):
		self.lc = lc
		if simplify:
			self.simplify()
	def simplify(self):
		new_lc = []
		self.lc.sort(key = lambda x : x.generator)
		for gen, group in itertools.groupby(self.lc, key = lambda x: x.generator):
			group = list(group)
			new_coeff = sum( [coeff for coeff, gen in group] )
			if new_coeff != 0:
				new_lc.append( Term(new_coeff, gen) )
		self.lc = new_lc
		return self

	def __add__(self, other):
		return LinearCombinationOfCD(copy.deepcopy(self.lc) + copy.deepcopy(other.lc), simplify = True)

	def __sub__(self, other):
		return LinearCombinationOfCD(copy.deepcopy(self.lc) + copy.deepcopy((-other).lc), simplify = True)
	
	def __pow__(self, other):
		if other == 1:
			return self
		if other % 2:
			exponent = (other-1)/2
			return self*(self)**exponent*(self)**exponent
		else:
			exponent = other/2
			return self**exponent * self**exponent
			
	
	def __neg__(self):
		return (-1)*self

	def __mul__(self, other):
		if isinstance(other, LinearCombinationOfCD):
			return LinearCombinationOfCD(
		 [Term(t1.coeff*t2.coeff, t1.generator*t2.generator) 
			for t1, t2 in itertools.product(self.lc, other.lc) ]
		  , simplify=True)
		elif iscoefficient(other):
			return LinearCombinationOfCD([ Term(other*c, g) for c, g in self.lc ], simplify=True)
	
	def __rmul__(self, other):
		if isinstance(other, LinearCombinationOfCD):
			return LinearCombinationOfCD(
		 [Term(t1.coeff*t2.coeff, t1.generator*t2.generator) 
			for t1, t2 in itertools.product(other.lc, self.lc) ]
		  , simplify=True)
		elif iscoefficient(other):
			return LinearCombinationOfCD([ Term(other*c, g) for c, g in self.lc ], simplify=True)

	def series(self, coefficients, order, n):
		for c, g in self.lc:
			if g.n != n:
				raise ValueError("The term %r does not have %i strands" % (str(g), n))
		res = coefficients[0]*LinearCombinationOfCD.unit(n)
		for i in range(1, order+1):
			res = res + coefficients[i]*(self**i)
		return res.up_to_order(order)
		
	def double(self, which):
		#Check this method
		def new_s(s, e, i, pos):
			if s < which:
				return (s, e)
			elif s == which:
				if i & (1 << pos):
					return (s+1, "placeholder"+str(e))
				else:
					return (s, "placeholder"+str(e))
			else:
				return (s+1, e)
		def double_chord(coeff, chord, which):
			n = len(chord.endpoints[which])
			res = []
			for i in range(2**n):
				newchord = ChordDiagram(chord.n+1, copy.deepcopy(chord.endpoints[:which]) 
							  + [[], [] ] + copy.deepcopy(chord.endpoints[which+1:]))
				for ii,strand in enumerate(newchord.endpoints):
					newchord.endpoints[ii] = [ new_s(s, e, i, e) for s, e in strand ]
				real_positions = dict()

				for pos,d in enumerate(chord.endpoints[which]):
					newchord.endpoints[new_s(which, 0, i, pos)[0]].append( new_s( d[0], d[1], i, d[1] )  )
					real_positions["placeholder"+str(pos)] = len(newchord.endpoints[new_s(which, 0, i, pos)[0]])-1
				for strand in newchord.endpoints:
					for i,d in enumerate(strand):
						s, e = d
						if isinstance(e, str):
							strand[i] = (s, real_positions[e])
				newchord.assert_valid()
				res.append(Term(coeff, newchord))
			return LinearCombinationOfCD(res)
		return sum([double_chord(coeff, chord, which) for coeff, chord in self.lc], LinearCombinationOfCD([]))	
		
	def coords(self, basis):
		coords = [0]*len(basis.keys)
		for c, g in self.lc:
			coords[ basis[g] ] = c
		return coords
	
	def close_off(self, which, where, sign=True):
		def new_(s, e, chord):
			if s < which:
				return (s, e)
			elif s == which:
				if which < where:
					return (where-1, chord.signature[where] + chord.signature[which] - e - 1)
				else:
					return (where, chord.signature[where] + chord.signature[which] - e - 1)
			else:
				return (s-1, e)
		res = []
		for coeff, chord in self.lc:
			new_chord = ChordDiagram(chord.n-1, copy.deepcopy(chord.endpoints))
			for i,strand in enumerate(new_chord.endpoints):
				new_chord.endpoints[i] = [ new_(s, e, chord) for s, e in new_chord.endpoints[i] ]
			new_chord.endpoints[where] +=  new_chord.endpoints[which][::-1] 
			new_chord.endpoints.pop(which)
			new_chord.assert_valid()
			if sign:
				res.append( Term( ((-1)**chord.signature[which])*coeff, new_chord ) )
			else:
				res.append( Term( coeff, new_chord  ))
				
		return LinearCombinationOfCD(res, simplify=True)

	def S(self, which):
		return LinearCombinationOfCD( [ Term((-1)**gen.signature[which]*coeff, gen.S(which)) for coeff, gen in self.lc] )
	
	def close_under(self, which, where):
		if where > which:
			new_where = where - 1
		else:
			new_where = where
		return self.S(which).S(where).close_off(which, where).S(new_where)

	def up_to_order(self, n):
		return LinearCombinationOfCD([Term(c, g) for c, g in self.lc if g.number_of_chords <= n ])

	def sort(self):
		self.lc.sort(key = lambda t: t.generator)
		return self

	def group_by_chords(self):
		if len(self.lc) == 0:
			return []
		res = defaultdict(lambda : [])
		for c, g in self.lc:
			res[g.number_of_chords].append(Term(c, g))
		r = [to_LC([])] * (max(res.keys()) + 1) 
		for i in res.keys():
			r[i] = to_LC(res[i])
		return r
	
	def operation_up_to_order(self, other, maxorder, operation):
		m1 = self.group_by_chords() + [to_LC([])] * (maxorder + 1) #hack
		m2 = other.group_by_chords()  + [to_LC([])] * (maxorder + 1)
		res = to_LC([])
		for i in range(maxorder + 1):
			for j in range(maxorder + 1 - i):
				res = res + operation(m1[i], m2[j])
		return res	
	
	def log_op_exp(self, maxorder, operation, n=-1):
		"""n is only after the operation"""
		if n == -1:
			#add error message
			n = self.lc[0].generator.n
		
		powers = LinearCombinationOfCD.powers
		exp1 = sum([t*(1/sympy.factorial(i)) for  i, t in  enumerate(powers(self, maxorder))], to_LC([]))
		opexp = operation(exp1) - LinearCombinationOfCD.unit(n)
		if len(opexp.lc) == 0:
			return to_LC([])
		lg = sum([ (-1)**i * t *sympy.Rational(1, i+1) for i, t in enumerate(powers(opexp, maxorder)[1:]) ], to_LC([]))
		return to_LC(lg).simplify()

	def close_off_logexp(self, which, where, maxorder):
		operation = lambda x : x.close_off(which, where)
		return self.log_op_exp(maxorder, operation, n = self.lc[0].generator.n - 1 )

	def close_under_logexp(self, which, where, maxorder):
		operation = lambda x : x.close_under(which, where)
		return self.log_op_exp(maxorder, operation, n = self.lc[0].generator.n - 1 )

	def double_logexp(self, which, maxorder):
		operation = lambda x : x.double(which)
		return self.log_op_exp(maxorder, operation, n = self.lc[0].generator.n + 1 )
	
	def om(self, other, maxorder):
		"""Optimized multiplication"""
		return self.operation_up_to_order(other, maxorder, lambda x, y : x*y)

	def ot(self, other, maxorder):
		"""Optimized tensor"""
		return self.operation_up_to_order(other, maxorder, lambda x, y : LinearCombinationOfCD.tensor(x, y))
	
	def changed_signs(self):
		"""Multiply each term by (-1)^number of chords"""
		res = []
		for coeff, chord in self.lc:
			res.append(Term( coeff*(-1)**chord.number_of_chords, chord ))
		return to_LC(res)

	@classmethod
	def tensor(cls, lc1, lc2):
		return cls(
		 [Term(t1.coeff*t2.coeff, ChordDiagram.tensor(t1.generator, t2.generator)) 
			for t1, t2 in itertools.product(lc1.lc, lc2.lc) ]
		  , simplify=True)
	
	@staticmethod
	def powers(x, maxorder, n=-1):
		if n == -1:
			n = x.lc[0].generator.n
		pwrs = [to_LC(LinearCombinationOfCD.unit(n))]
		for i in range(maxorder):
			pwrs.append(  pwrs[-1].om(x, maxorder) )
		return pwrs  
		
	@classmethod
	def bch(cls, lc1, lc2, maxorder, n=-1):
		lc1, lc2 = to_LC(lc1), to_LC(lc2)
		
		if n == -1:
			n = lc1.lc[0].generator.n
		powers = LinearCombinationOfCD.powers
		exp1 = sum([t*(1/sympy.factorial(i)) for  i, t in  enumerate(powers(lc1, maxorder))], to_LC([]))
		exp2 = sum([t*(1/sympy.factorial(i)) for  i, t in  enumerate(powers(lc2, maxorder))], to_LC([]))
		prexp = exp1.om(exp2, maxorder) - LinearCombinationOfCD.unit(n)
		if len(prexp.lc) == 0:
			return to_LC([])
		lg = sum([ (-1)**i * t * sympy.Rational(1, i+1) for i, t in enumerate(powers(prexp, maxorder)[1:]) ], to_LC([]))
		return to_LC(lg).simplify()
		
	@classmethod
	def tensor_bch(cls, lc1, lc2, maxorder, n=-1):
		lc1 = to_LC(lc1)
		lc2 = to_LC(lc2)
		
		if n == -1:
			n = lc1.lc[0].generator.n +  lc2.lc[0].generator.n
		powers = LinearCombinationOfCD.powers
		exp1 = sum([t*(1/sympy.factorial(i)) for  i, t in  enumerate(powers(lc1, maxorder))], to_LC([]))
		exp2 = sum([t*(1/sympy.factorial(i)) for  i, t in  enumerate(powers(lc2, maxorder))], to_LC([]))
		prexp = exp1.ot(exp2, maxorder) - LinearCombinationOfCD.unit(n)
		if len(prexp.lc) == 0:
			return to_LC([])
		lg = sum([ (-1)**i * t *sympy.Rational(1, i+1) for i, t in enumerate(powers(prexp, maxorder)[1:]) ], to_LC([]))
		return to_LC(lg).simplify()
	
	
	def log_inverse(self, maxorder, n=-1):
		return (-1)*self

	@classmethod
	def unit(cls, n):
		return cls([ Term(1, ChordDiagram(n, [ [] for i in range(n) ])) ])

	@classmethod
	def lie_word_to_chords(cls, word):
		def multiply_words(w1, w2):
			return [ (v1[0]*v2[0], v1[1]+v2[1]) for v1, v2 in itertools.product(w1, w2) ]
		def commute_words(w1, w2):
			return multiply_words(w1, w2) + multiply_words( [(-1, '')], multiply_words(w2, w1) )
		tree = ['root', None, None]
		current = None
		ended = False
		if len(word.strip()) == 1:
			return [(1, word_to_chord(word))]
		for c in word:
			assert (not ended)
			
			if c == ' ':
				pass
			elif c == 'x' or c == 'y':
				if current[1] is None:
					current[1] =c
				else:
					assert(current[2] is None)
					current[2] = c
	
			elif c == '[':
				if current is None:
					current = tree
				elif current[1] is None:
					current[1] = [current, None, None]
					current = current[1]
				else:
					assert(current[2] is None)
					current[2] = [current, None, None]
					current = current[2]
			elif c == ']':
				current = current[0]
				if current == 'root':
					ended = True
			else:
				raise ValueError('did not expect this symbol in a Lie word')
		assert(ended)
		def tree_to_word(tree):
			if len(tree) == 1:
				assert (tree in 'xy')
				return [(1, tree)]
			else:
				return commute_words( tree_to_word(tree[1]), tree_to_word(tree[2]) )
		res = tree_to_word(tree)
		d = dict()
		for coeff, word in res:
			if word in d:
				d[word] += coeff
			else:
				d[word] = coeff
		
		return cls([Term(d[word], ChordDiagram.word_to_chord(word)) for word in d])
def to_LC(a):
	if isinstance(a, ChordDiagram):
		return LinearCombinationOfCD([Term(1, a)])
	elif isinstance(a, Term):
		return LinearCombinationOfCD([a])
	elif isinstance(a, list):
		if len(a) == 0:
			return LinearCombinationOfCD([])
		elif isinstance(a[0], Term):
			return LinearCombinationOfCD(a)
		return LinearCombinationOfCD([Term(1, e) for e in a])
	elif isinstance(a, LinearCombinationOfCD):
		return a
	else:
		raise ValueError("Don't know how to convert " + str(a) + " to a linear combination")
	
class PDFOutput:
	def __init__(self, pagemargin=0.5, linelen=50, pagelen=1000, linemargin=1.0):
		text.set(text.LatexRunner)
		self.pagemargin = pagemargin
		self.linemargin = linemargin
		self.linelen = linelen
		self.pagelen = pagelen
		self.clear()
	def clear(self):
		self.pages = []
		self.current_line = []
		self.current_canvas = canvas.canvas()
		self.cvspos = 0
		self.currlen = 0
		self.currpagelen = 0
			
	@staticmethod
	def bbox_wh(bbox):
		try:
			w, h = bbox.width(), bbox.height()
		except ValueError:
			w, h = 0, 0
		return (w, h)
	
	def flush_line(self):
		line_to_draw = self.line_to_canvas()
		if self.cvspos - PDFOutput.bbox_wh(line_to_draw.bbox())[1] <= -self.pagelen:
			self.draw_page()
			self.current_canvas = canvas.canvas()
			self.cvspos = 0
		self.draw_line(line_to_draw)
		self.current_line = []
		self.currlen = 0
		
	def draw_unfinished(self):
		if len(self.current_line) != 0:
			self.flush_line()
		if PDFOutput.bbox_wh(self.current_canvas.bbox())[1] != 0 :
			self.draw_page()
	
	def write(self, filename):
		self.draw_unfinished()
		document.document(self.pages).writePDFfile(filename)
	def empty_circle(self,  margin = 0.2, textmargin =0.15, x=0, y=0, endpointsdistance = 0.5, lmargin=0.3, rmargin=0.3):
		c = canvas.canvas()
		r = 0.6
		y = y 
		x += textmargin

		c.stroke(path.circle(x, y, r))
		self.add_canvas(c, lmargin=lmargin, rmargin=rmargin)

	def draw_circle(self, chord,  margin = 0.2, textmargin =0.15, x=0, y=0, endpointsdistance = 0.5, lmargin=0.3, rmargin=0.3):
		chord.assert_valid()
		c = canvas.canvas()
		maxpoints = max( chord.signature + [1] )
		#centering:
		r = 0.6
		y = y 
		x += textmargin
		pairs = chord.to_pairs()
		nc = chord.number_of_chords

		c.stroke(path.circle(x, y, r))
		for a, b in pairs:
			angle1 = (a[1]+0.5)*2*pi/nc/2
			angle2 = (b[1]+0.5)*2*pi/nc/2
			x1 = x + r*cos(angle1)
			y1 = y + r*sin(angle1)
			x2 = x + r*cos(angle2)
			y2 = y + r*sin(angle2)
			dist = (x1-x2)**2 + (y1-y2)**2
			factor = 3 - 2*dist/(4*r**2)
			c.stroke(path.curve(x1, y1, (x+x1*factor)/(factor+1), 
										(y+y1*factor)/(factor+1), 
										(x+x2*factor)/(factor+1), 
										(y+y2*factor)/(factor+1),  x2, y2),[style.linestyle.dashed])
			c.fill(path.circle(x1, y1, 0.05))
			c.fill(path.circle(x2, y2, 0.05))
			
		self.add_canvas(c, lmargin=lmargin, rmargin=rmargin)

	def draw_chord(self, chord, margin = 0.2, textmargin =0.15, x=0, y=0, endpointsdistance = 0.5, lmargin=0.3, rmargin=0.3):
		chord.assert_valid()
		c = canvas.canvas()
		def draw_little_arrow(x, y, cvs, delta = 0.07, k = 0.6):
			cvs.stroke(path.line(x, y, x-k*delta, y -delta))
			cvs.stroke(path.line(x, y, x+k*delta, y -delta))
		maxpoints = max(max( chord.signature + [1] ), 1)
		#centering:
		y = y + (- (maxpoints-1)/2)*endpointsdistance
		x += textmargin
		
		for i in range(chord.n):
			c.stroke(path.line(i + x, 0-margin + y, i+x, (maxpoints-1+y)*endpointsdistance + margin))
			draw_little_arrow(i+x, y - margin/2, c)
		
		pairs = chord.to_pairs()
		for a, b in pairs:
			if a[0] == b[0]:
				centerx = a[0] + x
				centery = ((a[1] + b[1])/2.0)*endpointsdistance + y
				radius =( abs(a[1] - b[1])/2)*endpointsdistance
				scale = (2 - exp((1-2*radius)/10))/(2.5*radius)
				if a[0]==0:
					scale = -scale
				c.stroke(path.path(path.arc(centerx, centery, radius, 90, 270, )), 
				[trafo.scale(sx=scale, sy=1, x = centerx, y = centery),
				style.linestyle.dashed, color.rgb.black])
				c.fill(path.circle(centerx, centery + radius, 0.05))
				c.fill(path.circle(centerx, centery - radius, 0.05))
			else:
				x1 = a[0] +x
				y1 = a[1]*endpointsdistance +y
				x2 = b[0] +x
				y2 = b[1]*endpointsdistance +y
				c.stroke(path.line(x1, y1, x2, y2),[style.linestyle.dashed])
				c.fill(path.circle(x1, y1, 0.05))
				c.fill(path.circle(x2, y2, 0.05))
		self.add_canvas(c, lmargin=lmargin, rmargin=rmargin)

	def draw_text(self, txt, lmargin=0.3, rmargin=0.3):
		c = canvas.canvas()
		t = c.text(0, 0, txt, [text.halign.boxleft, text.size(sizename="Large")])
		self.add_canvas(c, lmargin=lmargin, rmargin=rmargin)
	
	def draw_coeff(self, coeff, first=False, lmargin=0.3, rmargin=0.3):
		if coeff == 1 and first:
			#There is an issue with the alignment of coefficients, see e.g.
				#~ out = chordlib.PDFOutput(pagelen=10000, linelen=4, pagemargin=1)
				#~ out.clear()
				#~ c=4
				#~ sim = load_simplifier(1, c, prefix = 'opt')
				#~ rels = sim.relations[(1, c)]
				#~ basis = rels.basis
				#~ print("Number of basis elements for %d chords is %d" %(c, len(basis.basis)))
				#~ for i in range(basis.n):
					#~ print(basis.basis[i])
					#~ out.draw_chords(2*to_LC(basis.basis[i]), circles=True, forcenotfirst=True)
			
				#~ out.write("basis6.pdf")
			

			return
		txt =  coeff_latex(coeff, first) 
		self.draw_text(txt, lmargin=lmargin, rmargin=rmargin)
	
	def draw_chords(self, chords, dosort=True, circles=True, forcenotfirst=False, margin=0.2, textmargin=0.15, lmargin=0.2, rmargin=0.2):
		if dosort:
			chords = to_LC(chords).sort()
		else:
			chords = to_LC(chords)
		if len(chords.lc) == 0:
			self.draw_coeff(0, True)
			return
		first = True
		for coeff, chord in chords.lc:
			if forcenotfirst:
				self.draw_coeff(coeff, False, lmargin=lmargin, rmargin=rmargin)
			else:
				self.draw_coeff(coeff, first, lmargin=lmargin, rmargin=rmargin)
			if circles and chord.n==1:
				self.draw_circle(chord, margin=margin, textmargin=textmargin, lmargin=lmargin, rmargin=rmargin)
			else:
				self.draw_chord(chord, margin=margin, textmargin=textmargin, lmargin=lmargin, rmargin=rmargin)
			first = False
	
	def add_canvas(self, cvs, lmargin=0.3, rmargin=0.3, width=-1):
		if width==-1:
			width = unit.tocm(PDFOutput.bbox_wh(cvs.bbox())[0])
		if self.currlen + lmargin + width + rmargin >= self.linelen:
			line_to_draw = self.line_to_canvas()
			if self.cvspos - PDFOutput.bbox_wh(line_to_draw.bbox())[1] <= -self.pagelen:
				self.draw_page()
				self.cvspos = 0
				self.current_canvas = canvas.canvas()
			self.draw_line(line_to_draw)
			self.current_line = []
			self.currlen = 0
			self.current_line.append((lmargin, rmargin, cvs))
			self.currlen += lmargin + width + rmargin
		else:
			self.current_line.append((lmargin, rmargin, cvs))
			self.currlen += lmargin + width + rmargin

	def draw_page(self):
		print("Drawing page")
		self.pages.append(document.page(self.current_canvas, bboxenlarge = self.pagemargin))
		
	def line_to_canvas(self):
		lcvs = canvas.canvas()
		x=0
		for lm, rm, c in self.current_line:
			w, h = PDFOutput.bbox_wh(c.bbox())
			x += lm
			lcvs.insert(c, [trafo.translate(x=x, y=0)])
			x += unit.tocm(w) + rm
		return lcvs
		
	def draw_line(self, line):
		self.current_canvas.insert(line, [trafo.translate(x=0, y=self.cvspos)])
		self.cvspos -= PDFOutput.bbox_wh(line.bbox())[1] + self.linemargin
		
def draw_chord(chord, canvas, margin = 0.2, textmargin =0.15, x=0, y=0, endpointsdistance = 0.5):
	chord.assert_valid()
	def draw_little_arrow(x, y, canvas,	delta = 0.07, k = 0.6):
		canvas.stroke(path.line(x, y, x-k*delta, y -delta))
		canvas.stroke(path.line(x, y, x+k*delta, y -delta))
	maxpoints = max( chord.signature + [1] )
	#centering:
	y = y + (- (maxpoints-1)/2)*endpointsdistance
	x += textmargin
	
	for i in range(chord.n):
		canvas.stroke(path.line(i + x, 0-margin + y, i+x, (maxpoints-1+y)*endpointsdistance + margin))
		draw_little_arrow(i+x, y - margin/2, canvas)
	
	pairs = chord.to_pairs()
	for a, b in pairs:
		if a[0] == b[0]:
			#drawing a circle
			centerx = a[0] + x
			centery = ((a[1] + b[1])/2.0)*endpointsdistance + y
			radius =( abs(a[1] - b[1])/2)*endpointsdistance
			scale = (2 - exp((1-2*radius)/10))/(2.5*radius)
			if a[0]==0:
				scale = -scale
			canvas.stroke(path.path(path.arc(centerx, centery, radius, 90, 270, )), 
			[trafo.scale(sx=scale, sy=1, x = centerx, y = centery),
			style.linestyle.dashed, color.rgb.black])
			canvas.fill(path.circle(centerx, centery + radius, 0.05))
			canvas.fill(path.circle(centerx, centery - radius, 0.05))
		else:
			#drawing a line
			x1 = a[0] +x
			y1 = a[1]*endpointsdistance +y
			x2 = b[0] +x
			y2 = b[1]*endpointsdistance +y
			canvas.stroke(path.line(x1, y1, x2, y2),[style.linestyle.dashed])
			canvas.fill(path.circle(x1, y1, 0.05))
			canvas.fill(path.circle(x2, y2, 0.05))
			

def draw_chords(chords, cvs, textmargin=0.15, linelen=20, linemargin = 0.2, dosort=True):
	if dosort:
		chords = to_LC(chords).sort()
	else:
		chords = to_LC(chords)
	if len(chords.lc) == 0:
		t = cvs.text( 0,0, '$0$', [text.halign.boxleft, text.size(sizename="Large")])
		return
	x = 0 
	y = 0
	y_abs = 0
	first = True
	linecanvas = canvas.canvas()
	line = 0
	for coeff, chord in chords.lc:
		if x > linelen:
			cvs.insert(linecanvas, [trafo.translate(x=0, y=y_abs)])
			x = 0
			y = 0
			y_abs -= linecanvas.bbox().height()*1.2 + linemargin
			linecanvas = canvas.canvas()
			line += 1
		txt =  coeff_latex(coeff, first) 
		t = linecanvas.text(x+textmargin, y, txt , [text.halign.boxleft, text.size(sizename="Large")])
		xnew= x + t.width +textmargin
		draw_chord(chord, linecanvas, x=xnew, y=y, textmargin=textmargin)
		x = xnew + max(chord.n -1, 1/2) + textmargin
		first = False

	cvs.insert(linecanvas, [trafo.translate(x=0, y=y_abs)])
	
def draw_equation(lhs, rhs, cvs, textmargin=0.15, linelen=20, linemargin = 0.2, dosort=True):
	if dosort:
		lhs = to_LC(lhs).sort()
		rhs = to_LC(rhs).sort()
	else:
		lhs = to_LC(lhs)
		rhs = to_LC(rhs)
	x = 0 
	y = 0
	y_abs = 0
	first = True
	linecanvas = canvas.canvas()
	line = 0
	if len(lhs.lc) == 0:
		t = linecanvas.text( 0,0, '$0$', [text.halign.boxleft, text.size(sizename="Large")])
	else:
		for coeff, chord in lhs.lc:
			if x > linelen:
				cvs.insert(linecanvas, [trafo.translate(x=0, y=y_abs)])
				x = 0
				y = 0
				y_abs -= linecanvas.bbox().height()*1.2 + linemargin
				linecanvas = canvas.canvas()
				line += 1
			txt =  coeff_latex(coeff, first) 
			t = linecanvas.text(x+textmargin, y, txt , [text.halign.boxleft, text.size(sizename="Large")])
			xnew= x + t.width +textmargin
			draw_chord(chord, linecanvas, x=xnew, y=y, textmargin=textmargin)
			x = xnew + max(chord.n -1, 1/2) + textmargin
			first = False
	t = linecanvas.text( x + textmargin, y, '=', [text.halign.boxleft, text.size(sizename="Large")])
	x = x + 2*textmargin + t.width
	if len(rhs.lc) == 0:
		t = linecanvas.text( x,y, '$0$', [text.halign.boxleft, text.size(sizename="Large")])
	else:
		for coeff, chord in rhs.lc:
			if x > linelen:
				cvs.insert(linecanvas, [trafo.translate(x=0, y=y_abs)])
				x = 0
				y = 0
				y_abs -= linecanvas.bbox().height()*1.2 + linemargin
				linecanvas = canvas.canvas()
				line += 1
			txt =  coeff_latex(coeff, first) 
			t = linecanvas.text(x+textmargin, y, txt , [text.halign.boxleft, text.size(sizename="Large")])
			xnew= x + t.width +textmargin
			draw_chord(chord, linecanvas, x=xnew, y=y, textmargin=textmargin)
			x = xnew + max(chord.n -1, 1/2) + textmargin
			first = False
	cvs.insert(linecanvas, [trafo.translate(x=0, y=y_abs)])
	
	

