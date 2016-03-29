import sympy
from table import (
    make_SI,
    write,
)

def error(f, err_vars=None):                # mit z.B. err_vars=(E,q) lassen sich die fehlerbehafteten Größen übermitteln.
    from sympy import Symbol, latex
    s = 0
    latex_names = dict()

    if err_vars == None:
        err_vars = f.free_symbols

    for v in err_vars:
        err = Symbol('latex_std_' + v.name)
        s += f.diff(v)**2 * err**2
        latex_names[err] = '\\sigma_{' + latex(v) + '}'

    return latex(sympy.sqrt(s), symbol_names=latex_names)

# D1, P, m, cw = sympy.var(r'D_1 P m c_w')
# mkck = sympy.var('mkck')
# vreal = D1/P*(m*cw+mkck)
# write('build/Fehlerformel_1.tex', error(vreal, err_vars=(D1,P, m, cw)))
#
# T1, T2 = sympy.var('T_1 T_2')
# videal = T1/(T1-T2)
# write('build/Fehlerformel_2.tex', error(videal, err_vars=(T1,T2)))
#
# dT, A, B, t = sympy.var(r'\td{T}{t} A B t')
# f = 2*A*t + B
# write('build/Fehlerformel_3.tex', error(f, err_vars=(A,B)))
