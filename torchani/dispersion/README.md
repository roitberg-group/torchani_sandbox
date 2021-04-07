# Some notes on D3 and dispersion corrections in ANI

## Original proposal
The original D3 (which I will call D3Zero)
is a DFT-D type method first proposed by Grimme et. al. in the paper:

A consistent and accurate ab initio parametrization of density functional
dispersion correction (DFT-D) for the 94 elements H-Pu
J. Chem. Phys. 132, 154104 (2010); https://doi.org/10.1063/1.3382344

D3Zero is just an energy correction term that is added on top of the E_ks functional
energy of the respective functional. In the original paper D3Zero is proposed with: 
- two body terms (R^6 and R^8)
- three body terms
- Zero damping function

Note that in practice *three body terms are neglected in all implementations by
default* orca and Grimme allow asking for the three body term by using the ABC
(or -abc in the case of Grimme's code) keywords.

## Functional form
The total correction has the form:

E_D3Zero = two_body_sum + three_body_sum 

where:

two_body_sum =  R^6 term + R^8 term
R^6 term = - sum_over_pairs(s6 * C6(r_ab) * zero_damp6(r_ab)/ r^6_ab)  
R^8 term = - sum_over_pairs(s8 * C8(r_ab) * zero_damp8(r_ab)/ r^8_ab)  

three_body_sum = -sum_over_triples(C9(r_abc) * cosine_term * zero_damp3B(r_mean_abc)/ (r_ab * r_bc * r_ca)^3 )
cosine_term = C9(r_ab, r_ac, r_bc) * (3 * cos th_a * cos th_b * cos th_c + 1)

r_mean_abc = (r_ab + r_bc + r_ca) / 3 (geometric mean)

the zero_damp function has two parameters, alpha and sr, and is given by

zero_damp(r_ab) = 1 / (1 + 6 ( sr * r_ab / R0_ab)^-alpha)

for R^6 the term zero_damp6 has a tunable parameter sr6 and alpha6 = 14
for R^8 the term zero_damp6 has a fixed sr8 = 1 and alpha8 = 16
for 3-body the term zero_damp3B has a fixed sr3B = 4/3 and alpha3B = 16

C6 is given by:
C6(cn(a, x), cn(b, x)) = sum_over_references(C6_ab_ref(cn_a_i, cn_b_j) Lij ) / sum_over_references(Lij)
Lij = exp(-k3 * (cn_a - cn_a_i)^2 + (cn_b - cn_b_j)^2 )
each pair of species a b has some number of references, between 1 and 5, and 
the parameters are calclated using precomputed parameters from the references, C6_ab_ref, 
and precomputed coordination numbers from the references, cn_a_i and cn_b_j
the constant k3 = 4 (fixed)

the continuous coordination numbers are given by:
cn(a, x) = sum_over_atoms_b_diff_a(1/(1 + exp(-k1(k2(R_a + R_b)/r_ab -1))))
where k1 = 16 and k2 = 4/3 are fixed, and R_j are the covalent radii

the coefficients C6 and C8 are obtained as follows:

C8(a, b) = 3 * C6(a, b) * sqrt( Qa * Qb )
where Qj = s42 * sqrt(Za) * <r^4>_a / <r^2>_b
s42 is fixed to 0.5 and <r^4>_a / <r^2>_a is precomputed for all species at 
a high level of theory, so Qa and Qb are essentially precomputed


The cutoff radii R0_ab are calculated for all pairs of atoms
and are given by

## Free parameters
Parameters for a family of functionals are calculated in this paper. wB97X is
*not* included in this family. 

There are four free parameters in this implementation, "sr6", "sr8", "s6" and
"s8".
- sr6 and sr8 are used for the damping function of the R^6 and R^8 two body terms, 
- s6 and s8 are used for the scaling of the R^6 and R^8 two body terms

"typically" s6 is set to 1.0 except in some double hybrids (which inherently
account for some R^6 interaction), in that case s6 < 1 "can happen".

sr8 is actually fixed to 1 for all functionals, this means they have 3 free
parameters, s6, sr6 and s8, of which s6 is usually exactly 1 (they only use 0.5
for B2PLYP), so in practice there are 2 free paramteres *sr6 and s8* of which
*the most important is sr6*

## BJ variation

In their paper 

Effect of the damping function in dispersion corrected density functional
theory Stefan Grimme  Stephan Ehrlich  Lars Goerigk JCC (2011)

Grimme et. al. propose a modification to D3Zero that uses a different damping
function, this damping function is termed "Becke-Johnson" or "BJ" or "Rational"
and is given by adding a term to the r^6 or r^8 denominator, instead of
multiplying the numerator by a term as in zero damping.

The two body terms then look like this:

R^6 term = - sum_over_pairs(s6 * C6(r_ab) * / (r^6_ab + bj_damp6(R0_ab)^6))
R^8 term = - sum_over_pairs(s8 * C8(r_ab) * / (r^8_ab + bj_damp8(R0_ab)^8))

where now bj_damp is given by

bj_damp(R0_ab) = a0 * R0_ab + a1
where a1 and a0 are free parameters to optimize, and R'0_ab are cutoff radii 
that are determined differently from the cutoff radii in the zero_damp function.
these cutoff radii are given by:
R'0_ab = sqrt(C8_ab/C6_ab) = sqrt(3 * sqrt(Qa * Qb)) = sqrt(3) * fourth_root(Qa * Qb)

s8 has the same physical meaning in D3Zero and D3BJ
sr6 and a1 have the same physical meaning also
a2 is an extra parameter for fine tuning

## D3MZero and D3MBJ by Sherrill et. al. (TODO)

Basically a refit in the case of D3MBJ and adds an extra beta parameter in
the case of D3MZero

# Notes about \omegaB97X and related functionals

## original wB97X/wB97 work
wB97 and wB97X were first proposed in the paper 

Systematic optimization of long-range corrected hybrid density functionals
J. Chem. Phys. 128, 084106 (2008) Jeng-Da Chai and Martin Head-Gordon

The idea behind this functionals is to model "long range corrections", so 
they are termed "LC hybrids"

These functionals are hybrid, so their energy is given by 
E_xc = c_x * E_x_hf  + E_x_df

This means that they incorporate part of the exchange interaction coming from 
Hartree Fock, using a coefficient c_x, typically c_x ~ 0.2-0.25

They propose the parametrization:

E_xc = E_x_hf(LR) + c_x * E_x_hf(SR) + E_x_df(SR) + E_c_df(LR), for wB97X
(the X stands for the c_x term)

or 

E_xc = E_x_hf(LR) + E_x_df(SR) + E_c_df(LR), for wB97

where LR and SR stand for long range and short range respectively

The idea is that the contributions from both Hartree Fock and DF are done by
weighting differently the long range and short range parts of the potential.
and the weighting is done using a splitting erf operator in the coulomb kernel
so that instead of integrating 1/r one integrates 

erf(omega * r)/r + erfc(omega * r)/r 

where erf and erfc are the error function and complementary error function

(I'm not sure about this) for E_x_df(SR) they use the local density
approximation (LSDA), it is possible to use LSDA for long range also, in which
case they term the functional **RSHXLDA** but this is not much used in
chemistry

for their fitting they explore a limited range of omega values, 0-0.5 in steps
of 0.1 (Bohr^-1)

They truncate functional expansions (?) at order 4, this gives them 
- 15 parameters plus omega, plus Cx, for both density functionals
In the limit when omega -> 0, the forms of wB97X is equal to B97 and wB97 is
equal to HCTH, (but with different fitted parameters)

## Dispersion corrected wB97X-D (or more clearly, wB97X'+D2mod)
This functional was proposed in the paper 

Long-range corrected hybrid density functionals with damped atom–atom
dispersion corrections Jeng-Da Chai  and  Martin Head-Gordon (2008)

In this paper it is proposed to use the same functional form as wB97X, and adding
dispersion corrections of the "D2 form with modifications" proposed by Grimme.
I will call this modified D2 dispersion D2mod
They set the D2mod parameter s6 to 1.0 (D2 in general fits this parameter
differently for different functionals, they don't fit it), and they 
use a different damping function than D2. Their damping function is 
given by:

zero_dampD2mod = 1/(1 + a(r_ab/(R_vdw_a + R_vdw_b))^-alpha)

where in this case they use 12 instead of 14 for the alpha parameter, they also
fit a to 6.0, which is the exact value Grimme uses with zero_damp in D3Zero.

Instead of just adding the D2mod energy, they **fully refit all 17 parameters of
the functional**, this means this functional can not be used without the D2mod
correction in any way. Since this functional is refitted I will call the 
refitting wB97X'

Gaussian16 only has these three variants and no others, of the wB97X functional: 
- wB97
- wB97X
- wB97X-D (wB97X'+D2mod, and the Gaussian keyword is wB97XD to make things more confusing)

Orca includes only:
- wB97
- wB97X

## Dispersion corrected wB97X-D3Zero (or more clearly wB97X''+D3Zero)

In their paper
Long-Range Corrected Hybrid Density Functionals with Improved Dispersion Corrections
You-Sheng Lin, Guan-De Li, Shan-Ping Mao, and Jeng-Da Chai

(Note that Chai is one of the original authors of the wB97X paper and wB97X-D
paper)

two new functionals are proposed. They call the functionals wM06-D3 and wB97X-D3
They would be better termed wM06'+D3Zero  ans wB97X''+D3Zero. I will focus only
on the wB97X variant and not the minessotta variant.
What they actually do is a **further refit of all 17 parameters of the functional**
but this time adding parameters for Grimme's et. al. D3Zero also. Once again
this functional can't be used without the dispersion correction

They find that wB97X''+D3Zero performs better than wB97X''+D2mod for non bonded
interactions, but similar for bonded interactions.
The D3Zero parameters they find are sr6 = 1.281 and sr8 = 1.094, 
and they fix s6 = 1 and s8 = 1, so the procedure is slightly different to the
usual Grimme procedure, where s8 is variable instead of sr8.

Orca includes this functional with the keyword:
- wB97X-D3 (wB97X''-D3Zero)

## Evolutionary optimized, nonlocal correlated functional wB97X-V, using VV10 (more clearly wB97X_evo+VV10)

VV09 and VV10 is a different way to introduce nonlocal effects (i.e.
dispersion) that depends on the electron density. It is more computationally
expensive than D3.  VV09 has one optimizable parameter, and VV10 has 2
parameters.


In their paper, 

ωB97X-V: A 10-parameter, range-separated hybrid, generalized gradient
approximation density functional with nonlocal correlation, designed by a
survival-of-the-fittest strategy(Physical Chemistry Chemical Physics) (2014)
Narbe Mardirossiana and Martin Head-Gordon

Head Gordon et. al. introduce a different functional with dispersion corrections.

Instead of using the old functional form for wB97X in this paper they 
use a sort of evolutionary algorithm to choose better parameters to optimize from 
the expansion series instead of using some truncation after a given order. 
As a result the functional actually has a different number 
of parameters than wB97, wB97X, wB97X'+D2Mod and wB97X''+D3Zero. 
This functional has 7 linear parameters, plus omega and both VV10 parameters.
Since the actual functional form is different this time, I will term this 
functional wB97X_evo, (for evolutionary) which means in my notation this is
wB97X_evo+VV10

Orca includes this functional with the keyword:
- wB97X-V (wB97X_evo+VV10)

## Combinatorially optimized functional, B97M-V, using VV10 (more clearly B97X_com+VV10)

Very similar to the wB97M-V (wB97X_com+VV10), but in a previous paper.
Functional form a bit different, have not read the paper in depth but the
combinatorial procedure is analogous

Proposed in the 2015 paper:
Mapping the genome of meta-generalized gradient approximation density
functionals: The search for B97M-V J. Chem. Phys. 142, 074111 (2015);
https://doi.org/10.1063/1.4907719
Narbe Mardirossian1 and Martin Head-Gordon

Orca includes this functional with the keyword:
- B97M-V (B97X_com+VV10)

## Other combinatorially optimized, nonlocal correlated functional wB97M-V, using VV10 (more clearly wB97X_com+VV10)

Essentially similar to what is done in the wB97X_evo+VV10 paper, but using 
a different, combinatorial optimization. The relevant paper is:

wB97M-V: A combinatorially optimized, range-separated hybrid, meta-GGA density
functional with VV10 nonlocal correlation Narbe Mardirossian1 and Martin
Head-Gordon (Journal of Chemical Physics) (2016)

I will name this functional wB97X_com+VV10 in analogy with wB97X_evo+VV10

Orca includes this functional with the keyword:
- wB97M-V (wB97X_com+VV10)

## Yet another combinatorially optimized ... etc 

Essentially similar to what is done in the wB97X_evo+VV10 paper, but using 
a different, combinatorial optimization. The relevant paper is:

Survival of the most transferable at the top of Jacob’s ladder: Defining and
testing the wB97M(2) double hybrid density functional featured J. Chem. Phys.
148, 241736 (2018); https://doi.org/10.1063/1.5025226 

I will name this functional wB97X_com2+VV10 in analogy with wB97X_com+VV10

Orca does not include this functional.


## D3BJ and D3(0)corrected functionals wB97X-D3(BJ) and wB97X-D3(BJ) (and others) (more clearly wB97X_evo+D3BJ and wB97X_com+D3BJ)

In their 2018 paper:

The Nonlocal Kernel in van der Waals Density Functionals as an
Additive Correction: An Extensive Analysis with Special Emphasis on
the B97M‑V and ωB97M‑V Approaches (2018), JCTC
Asim Najibi and Lars Goerigk

Goerigk et. al. Perform fittings of D3BJ parameters for the exact functional
forms and parameters of wB97_evo and wB97_com (without the VV10 nonlocal term).
Their idea was to fully replace the VV10 nonlocal term using a D3BJ term. 
In some cases their D3BJ functionals outperform the VV10 functionals.
The parameters they found are, for D3BJ:

functional,    s6,   a1,    s8,    a2
wB97X_evo+D3BJ 1.0000 0.0000 0.2641 5.4959
wB97X_com+D3BJ 1.0000 0.5660 0.3908 3.1280
B97X_com+D3BJ  1.0000 -0.0780 0.1384 5.5946

and for  D3Zero:

functional s6,     sr6,        s8
B97X_com+D3Zero  1.0000 0.9342 -0.2582
wB97X_com+D3Zero 1.0000 1.5072 1.5148

Unfortunately, even though they keep the exact functional form of the wB97X_evo
and wB97X_com functionals, they change the nomenclature once more and call the 
functionals with D3, wB97X-D3(BJ) and wB97M-D3(BJ), they also use the D3(0) 
notation for D3Zero

They also fit the functional B97M-V (B97X_com+VV10) to both D3BJ and D3Zero, and they fit the
functional wB97X_com to D3Zero also. I'm not sure why they don't fit wB97_evo to D3Zero.
 
Orca includes this functionals with the keywords:
- wB97X-D3BJ (wB97X_evo+D3BJ)
- wB97M-D3BJ (wB97X_com+D3BJ)
- B97M-D3BJ  (B97X_com+D3BJ)

## Summary

As a summary the following functionals are available in orca and gaussian, and
they have the following "nonlocal" corrections (I will put D3 and VV10 in the
same bag here):

Functional name , paper authors, year, density functional part, Orca keyword, Gaussian16 keyword, Nonlocal correction, Fitted with NL term?, Relevant DOI

wB97              Chai+HGor_1  2008             wB97           wB97         wB97                  None                  No                       10.1039/b810189b
wB97X             Chai+HGor_1  2008             wB97X          wB97X        wB97X                 None                  No                       10.1039/b810189b
wB97X-D           Chai+HGor_2  2008             wB97X'         -            wB97XD                D2mod                 yes, with D2mod          10.1039/b810189b
wB97X-D3          Sheng+Chai   2013             wB97X''        wB97X-D3       -                   D3Zero                yes, with D3Zero         10.1021/ct300715s
wB97X-V           Mard+HGor_1  2014             wB97X_evo      wB97X-V        -                   VV10                  yes, with VV10           10.1039/C3CP54374A
B97M-V            Mard+HGor_2  2015             B97X_com       B97M-V         -                   VV10                  yes, with VV10           10.1063/1.4907719
wB97M-V           Mard+HGor_3  2016             wB97X_com      wB97M-V        -                   VV10                  yes, with VV10           10.1063/1.4952647
wB97M(2)          Mard+HGor_4  2018             wB97X_com2      -             -                   VV10                  yes, with VV10           10.1063/1.5025226 
wB97X-D3(BJ)      Naj+Goer     2018             wB97X_evo      wB97X-D3BJ     -                   D3BJ                  yes, with VV10           10.1021/acs.jctc.8b00842
wB97M-D3(BJ)      Naj+Goer     2018             wB97X_com      wB97M-D3BJ     -                   D3BJ                  yes, with VV10           10.1021/acs.jctc.8b00842
B97M-D3(BJ)       Naj+Goer     2018             B97X_com       B97M-D3BJ      -                   D3BJ                  yes, with VV10           10.1021/acs.jctc.8b00842
wB97M-D3(0)       Naj+Goer     2018             wB97X_com       -             -                   D3Zero                yes, with VV10           10.1021/acs.jctc.8b00842
B97M-D3(0)        Naj+Goer     2018             B97X_com        -             -                   D3Zero                yes, with VV10           10.1021/acs.jctc.8b00842


Note that orca actually spits out some energy + some dispersion if you use the
keywords "wB97X D2", but it also outputs a warnings "non-parametrized
functional used for VDW correction" which basically means it uses some made up
D2 parameters.

In this table primes denote refittings of the same functional form.

Note that all functionals except for wB97X and wB97 were fitted in such a way
as to simultaneously optimmize the functional parameters and some nonlocal
correlation parameters. In the case of wB97X_evo+D3BJ and wB97X_com+D3BJ the
nonlocal correlation **was then replaced with a different one** (the comments
on the last column are correct, the functional form is the same, and the
parameters of the functional form were originally optimized with VV10), but the
point still is that using the functional without any nonlocal correlation is
**not justified under any means** for all functionals except wB97 and wB97X.
On the other hand, there are no fitted nonlocal correlation parameters for the
original wB97 and wB97X, so if a nonlocal correlation is needed the parameters
should be fitted from scratch, however, it is very possible that the results
would be suboptimal, since from the paper's conclusions it seems that the
functional has to be reoptimized together with the D3 parameters in order for
the parameters to make sense.


## TODO

- Sherrill et al paper on D3MBJ / D3MZero 
- Paper on D2   https://doi.org/10.1002/jcc.20495 ("D1" doesn't really exist?)
- Paper on D4 (2017); https://doi.org/10.1063/1.4993215 although looks like it
  performs some numerical integration using mulliken charges, hard for torch.
- Paper on VV10 and VV09, and other dispersion like interactions
