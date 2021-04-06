# Some notes on D3 and dispersion corrections in ANI

D3 is a DFT-D type method first proposed by Grimme et. al. in the paper:

A consistent and accurate ab initio parametrization of density functional
dispersion correction (DFT-D) for the 94 elements H-Pu
J. Chem. Phys. 132, 154104 (2010); https://doi.org/10.1063/1.3382344

D3 is just an energy correction term that is added on top of the E_ks functional
energy of the respective functional. In the original paper D3 is proposed with: 
- two body terms (R^6 and R^8)
- three body terms
- Zero damping function


The total correction has the form:

E_D3 = two_body_sum + three_body_sum 

where:

two_body_sum =  R^6 term + R^8 term
R^6 term = - sum_over_pairs(s6 * C6(r_ab) * zero_damp6(r_ab)/ r^6_ab)  
R^8 term = - sum_over_pairs(s8 * C8(r_ab) * zero_damp8(r_ab)/ r^8_ab)  

three_body_sum = -sum_over_triples(C9(r_abc) * cosine_term * zero_damp3B(r_mean_abc)/ (r_ab * r_bc * r_ca)^3 )
cosine_term = C9(r_ab, r_ac, r_bc) * (3 * cos th_a * cos th_b * cos th_c + 1)

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



Parameters for a family of functionals are calculated in this paper. wB97X is
*not* included in this family. The functionals are: 
- B2PLYP
- PW6B95
- B97-D
- revPBE
- B3LYP
- BLYP
- TPSS0
- PBE0
- TPSS
- PBE
- BP86

There are four free parameters in this implementation, "sr6", "sr8", "s6" and "s8".
- sr6 and sr8 are used for the damping function of the R^6 and R^8 two body terms, 
- s6 and s8 are used for the scaling of the R^6 and R^8 two body terms

"typically" s6 is set to 1.0 except in some double hybrids (which inherently
account for some R^6 interaction), in that case s6 < 1 "can happen".

sr8 is actually fixed to 1 for all functionals in their paper, this means they
have 3 free parameters, s6, sr6 and s8, of which s6 is usually exactly 1 (they
only use 0.5 for B2PLYP)













