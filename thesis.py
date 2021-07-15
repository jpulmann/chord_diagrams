import chordlib
from chordlib import LinearCombinationOfCD, to_LC, function_to_coeffs, ChordDiagram, Term

import elimination
from generate_basis import generate, load_simplifier
import numpy as np
import sympy
from pyx import *

#The maximal number of chords to consider, for P_1, P_2 ...
max_chords = (5, 4, 4, 4, 3)

#Change to True after running the script once, to avoid regenerating the basis every time
ALREADY_GENERATED_BASIS = False

#This prefix allows to distinguish different bases.
BASIS_PREFIX = "b"

def KZAssociator(maxorder):
    """ Returns the logarithm of KZ associator, maximal maxorder possible is 4. See page 299 of [CMD] """
    Pi = sympy.symbols("\\pi")
    Apery = sympy.symbols("\\zeta(3)")

    second_order = sympy.Rational(1, 24)*LinearCombinationOfCD.lie_word_to_chords('[xy]')
    third_order = (-sympy.I*Apery/Pi**3*sympy.Rational(1, 8))* ( 
                    LinearCombinationOfCD.lie_word_to_chords('[x[xy]]') + 
                    LinearCombinationOfCD.lie_word_to_chords('[y[xy]]'))    
                    
    fourth_order = -sympy.Rational(1, 16*360) * ( 
                    4*LinearCombinationOfCD.lie_word_to_chords('[x[x[xy]]]') + 
                    4*LinearCombinationOfCD.lie_word_to_chords('[y[y[xy]]]') + 
                    LinearCombinationOfCD.lie_word_to_chords('[y[x[xy]]]')) 
    lPhi = (second_order + third_order + fourth_order).up_to_order(maxorder)
    return lPhi

def Twist(assoc, twist, maxorder):
    """Twist the associator 'assoc' using 'twist' up to maxorder  """
    print("Twisting")
    twist = twist.up_to_order(maxorder)
    assoc = assoc.up_to_order(maxorder)
    
    twistinv = (twist-units[1]).series(function_to_coeffs(1/(1+x), x, maxorder), maxorder, 2).up_to_order(maxorder)
    T1 = tensor(units[0], twist)
    T2 = twist.double(1)
    T3 = twistinv.double(0)
    T4 = tensor(twistinv, units[0])
    return ((T1*T2).up_to_order(maxorder)*(assoc*(T3*T4).up_to_order(maxorder)).up_to_order(maxorder)).up_to_order(maxorder)
    

if __name__ == "__main__":
    #Initializing an empty PDF to output chord diagrams
    out = chordlib.PDFOutput(pagelen=40, linelen=20)
    
    if not ALREADY_GENERATED_BASIS:
        print("Generating bases")
        for n,m in enumerate(max_chords):
            #Generating a basis of P_n up to m chords, with both optimizations
            generate(n+1, m, prefix = BASIS_PREFIX)
            
    #Load the basis, in the form of a "simplifier"
    print("Loading bases")
    sim = load_simplifier(1, max_chords[0], prefix = BASIS_PREFIX)
    for n, m in enumerate(max_chords[1:]):
        sim = load_simplifier(n+2, m, prefix = BASIS_PREFIX, simplifier=sim)
    
    ###Check that the Pentagon equation for the associator holds
    maxorder = 4
    tensor = LinearCombinationOfCD.tensor
    units = [LinearCombinationOfCD.unit(i) for i in range(1, 6)]
    lPhi = KZAssociator(maxorder)
    print("Calculating the exponentials of the logarithm of the associator and its inverse")
    x = sympy.Symbol('x')
    Phi = lPhi.series(function_to_coeffs(sympy.exp(x), x, maxorder), maxorder, 3)
    Phiinv = (-lPhi).series(function_to_coeffs(sympy.exp(x), x, maxorder), maxorder, 3)
    
    out.draw_text('The KZ associator is:')
    out.flush_line()
    out.draw_chords(Phi)
    out.flush_line()
    
    out.draw_text('Pentagon up to order %i' % maxorder)
    lhs = sim.simplify(((tensor(units[0], Phi)*Phi.double(1)).up_to_order(maxorder)*tensor(Phi, units[0])).up_to_order(maxorder))
    rhs = sim.simplify((Phi.double(2)*Phi.double(0)).up_to_order(maxorder))
    print("The difference of the two sides of the pentagon equation is equal to ", sim.simplify(lhs-rhs).lc)
    out.flush_line()
    out.draw_chords( lhs )
    out.draw_text('$- ($')
    out.draw_chords( rhs )
    out.draw_text('$ )=$')
    out.draw_chords(sim.simplify(lhs-rhs))
    out.flush_line()
    
    ###Compare our \nu with the one in [Chmutov, Duzhin and Mostovoy] on page 345:
    #b1i is the set of all chord diagrams (before quotienting) on 1 strand with i chords 
    b12 = sim.relations[(1, 2)].basis.basis
    b14 = sim.relations[(1, 4)].basis.basis
    nuinvCDM = units[0] - sympy.Rational(1, 24)*to_LC([Term(1, b12[0]), Term(-1, b12[1])]) + sympy.Rational(1, 5760)* to_LC([
        Term(7, b14[38]),
        Term(-17, b14[0]),
        Term(14, b14[3]),
        Term(-8, b14[9]),
        Term(3, b14[7]),
        Term(1, b14[39]),
        ]
    )
    out.draw_text('$\\nu^{-1} = $ from the [CMD] book on page 345 is')
    out.flush_line()
    out.draw_chords(nuinvCDM.up_to_order(maxorder), dosort=False)
    out.flush_line() 
    out.draw_text('the same element, written in our basis, is')
    out.flush_line()
    out.draw_chords(sim.simplify(nuinvCDM.up_to_order(maxorder)), dosort=False)
    out.flush_line()     
    #calculate nu^-1 from the KZ associator
    nuinvKZ = sim.simplify(Phi.close_under(1, 0).S(1).close_under(1, 0))
    out.draw_text('$\\nu^{-1} = $ from the the KZ associator')
    out.flush_line()
    out.draw_chords(nuinvKZ.up_to_order(maxorder), dosort=False)
    out.flush_line()
    out.draw_text('Their difference:')
    out.draw_chords(sim.simplify(nuinvCDM.up_to_order(maxorder)-nuinvKZ.up_to_order(maxorder)))
    out.flush_line()
    print("The difference of the two elements nu^-1 is ", sim.simplify(nuinvCDM.up_to_order(maxorder)-nuinvKZ.up_to_order(maxorder)).lc)

    ###Check that r3(\Phi) - 1 = 0
    PhiOnInvariants = sim.simplify(Phi.double(2)
        .close_under(2, 1).close_under(2, 0).up_to_order(4)
        - units[1])
    print("Phi acting on invariants minus 1 is equal to ", PhiOnInvariants.lc)
    out.write('results_maxorder4.pdf')

    ###Let us turn to the element a
    out.draw_text('Calculate a:')
    out.flush_line() 
    a = sim.simplify((tensor(units[0], Phiinv)*Phi.double(2)).up_to_order(maxorder)\
                                                .close_off(2, 1, maxorder)\
                                                .close_off(2, 0, maxorder))
    out.draw_text('$a \\quad = $')
    out.draw_chords(sim.simplify(a))
    out.flush_line()
    
    

    ###Outputting a up to order 3 to a separate file
    out_for_a = chordlib.PDFOutput(pagelen=50, linelen=24)
    out_for_a.draw_text('$a \\quad = $')
    out_for_a.draw_chords(sim.simplify(a.up_to_order(3)), textmargin=0.05)
    out_for_a.write('a.pdf')
    
    ###Checking Proposition 4.13:
    print('Difference between r_2(a) and nu^-1 is', sim.simplify( (a.close_under(1, 0) - nuinvKZ).up_to_order(maxorder) ).lc)
    
    ###Checking Proposition 4.14:
    print('a minus the first chord diagram in equation (4.3) in Proposition 4.14 is ', sim.simplify((a - Phiinv.close_off(0, 1))
        .up_to_order(maxorder)).lc)


    ###Twisting by a. 
    out.draw_text('Associator in the choses basis is:')
    out.flush_line() 
    out.draw_text('$\Phi \\quad = $')
    out.draw_chords(sim.simplify(Phi.up_to_order(maxorder)))
    out.flush_line()
    Phia = sim.simplify(Twist(Phi, a, maxorder))
 
    out.draw_text('Associator twisted by a, in the choses basis, is:')
    out.flush_line() 

    out.draw_text('$\Phi^a \\quad = $')
    out.draw_chords(Phia.up_to_order(maxorder))
    out.flush_line()
    print("The difference between Phi^a and Phi_-t is ", sim.simplify((Phia-Phi.changed_signs()).up_to_order(maxorder)).lc)
    

    out.write('results_maxorder4.pdf')

