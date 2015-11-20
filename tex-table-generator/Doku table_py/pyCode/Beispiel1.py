write('build/Tabelle_b_texformat.tex', make_full_table(
    'Messdaten Kapazitätsmessbrücke.',
    'table:b',
    'build/Tabelle_b.tex',
    [1,2,3,4,5],
    ['Wert', '$C_2 \\:/\\: \\si{\\nano\\farad}$', '$R_2 \\:/\\: \\si{\\ohm}$',
    '$R_2 / R_4$', '$R_x \\:/\\: \\si{\\ohm}$',
    '$C_x \\:/\\: \\si{\\nano\\farad}$']))
