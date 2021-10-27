from pint import UnitRegistry as UR
ur = UR()

m = 1 * ur['angstrom**3']
print(m.to(ur['cm**3']))
