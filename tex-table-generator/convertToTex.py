from table import (
    make_table,
    make_SI,
    write,
)

spalte_1=[1.7,2.3,3.5,4.4]
spalte_2=[10,20,30,40]

write('table.tex',
      make_table([spalte_1,spalte_2], [1,0]))
