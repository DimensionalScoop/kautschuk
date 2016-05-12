write('build/Tabelle_b_texformat.tex', make_full_table(
    caption = 'Messdaten Kapazitätsmessbrücke.',
    label = 'table:A2',
    source_table = 'build/Tabelle_b.tex',
    stacking = [1,2,3,4,5],                 # default = None
    units = ['Wert',                        # default = None
    r'$C_2 \:/\: \si{\nano\farad}$',
    r'$R_2 \:/\: \si{\ohm}$',
    r'$R_3 / R_4$', '$R_x \:/\: \si{\ohm}$',
    r'$C_x \:/\: \si{\nano\farad}$'],
    replaceNaN = True,                      # default = false
    replaceNaNby = 'not a number'))         # default = '-'
