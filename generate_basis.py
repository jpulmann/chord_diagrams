import chordlib
from chordlib import LinearCombinationOfCD, to_LC, function_to_coeffs, ChordDiagram, Term

import elimination

def raw_relations(n, c, verbose=True):
	if verbose:
		print('Generating all relations for n=%i and %i chords' % (n, c))
	all_diagrams = [cd for cd in chordlib.ChordDiagram.all_diagrams(n, c) if (not (0 in cd.signature))]
	new_all_diagrams = []
	for cd in all_diagrams:
		if cd.move_small_loops() == cd:
			new_all_diagrams.append(cd)
	all_diagrams = new_all_diagrams
	if verbose: print("Have %i diagrams" % len(all_diagrams))
	basis = chordlib.Basis(all_diagrams)
	if verbose: print("Have Basis")
	all_4T_relations = sum([diag.all4T() for diag in all_diagrams] , [])
	if verbose: print("Have %i 4T relations" % len(all_4T_relations))
	m = basis.to_matrix( all_4T_relations )
	return basis, m
	
def eliminate(basis, m, verbose=True):
	em, perm = elimination.eliminate(m)
	basis = basis.permute(perm)
	crelations = chordlib.Relations(basis, em)

	relations, total = crelations.rels, crelations.total
	print('For P_%i with %i chords, there is %i independent diagrams from %i' % (n, c, total-relations, total))
	simp = chordlib.CDSimplifier()
	simp.add_relations(chordlib.Relations(basis, em), n, c)
	return simp


def generate(n, maxc, prefix='', ES=True, SL=True):
	for c in range(0,maxc+1):
		print('-------------------')
		print(f"Generating a basis of P_{n} with {c} chords")
		if ES:
			all_diagrams = [cd for cd in chordlib.ChordDiagram.all_diagrams(n, c) if (not (0 in cd.signature))] 
		else:
			all_diagrams = [cd for cd in chordlib.ChordDiagram.all_diagrams(n, c)]# if (not (0 in cd.signature))] 
		if SL:
			new_all_diagrams = []
			for cd in all_diagrams:
				if cd.move_small_loops() == cd:
					new_all_diagrams.append(cd)
			all_diagrams = new_all_diagrams
		print("There are %i diagrams" % len(all_diagrams))
		basis = chordlib.Basis(all_diagrams)
		all_4T_relations = sum([diag.all4T(SLoptimization=SL) for diag in all_diagrams] , [])
		print("There are %i 4T relations" % len(all_4T_relations))
		m = basis.to_matrix( all_4T_relations )
		em, perm = elimination.eliminate(m)
		basis = basis.permute(perm)
		crelations = chordlib.Relations(basis, em)

		relations, total = crelations.rels, crelations.total
		print('For P_%i with %i chords, there is %i independent diagrams from %i' % (n, c, total-relations, total))
		#For saving the actual matrix with relations, import numpy and uncomment the next line:
		#~ numpy.savetxt('matrix.txt', em, fmt='%i')
		simp = chordlib.CDSimplifier()
		simp.add_relations(chordlib.Relations(basis, em), n, c)
		
		simp.save_relation('bases_data/%sd%i_%i.dat' % (prefix, n, c), n, c)

def load_simplifier(n, maxc, prefix='', simplifier=None, ESoptimization=True, SLoptimization=True):
	if simplifier is None:
		simplifier = chordlib.CDSimplifier(ESoptimization=ESoptimization, SLoptimization=SLoptimization)
	for c in range(0, maxc+1):
		simplifier.load_relation('bases_data/%sd%i_%i.dat' % (prefix, n, c), n, c)
	return simplifier

