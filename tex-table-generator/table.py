import itertools
import codecs
import uncertainties
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (
    nominal_values as noms,
    std_devs as stds,
)
from uncertainties import ufloat

def make_table(columns, figures=None):
    assert hasattr(columns[0],'__iter__'), "Wenn nur eine Zeile von Daten vorliegt, funktioniert zip nicht mehr; die Elemente von columns müssen Listen sein, auch wenn sie ihrerseits nur ein Element enthalten."

    if figures is None:
        figures = [None] * len(columns)

    cols = []
    for column, figure in zip(columns, figures):
        if np.any(stds(column)):
            if figure is None:
                figure = ''
            col = list(zip(*['{0:.{1:}uf}'.format(x, figure).split('+/-') for x in column]))
        else:
            col = list(zip(*[['{0:.{1:}f}'.format(x, figure)] for x in noms(column)]))
        cols.extend(col)

    max_lens = [max(len(s) for s in col) for col in cols]
    cols = [['{0:<{1:}}'.format(s, ml) for s in col] for col, ml in zip(cols, max_lens)]

    rows = list(itertools.zip_longest(*cols))

    return (r' \\' + '\n').join([' & '.join(s for s in row if s is not None) for row in rows]) + r' \\'

def make_composed_table(tables):
    assert isinstance(tables, list), "You need to give a list of filenames to make_composed_table!"
    Output = ''
    for filename in tables:
        with open(filename, 'r') as f:
            Output += f.read()
    return Output

def make_SI(num, unit, exp='', figures=None):
    y = ufloat(0.0, 0) #siunitx mag kein 0 +- 0, deshalb hier der workaround
    if num == y:
        return "(0 \pm 0) ~ \si{" + unit + "}"
    if np.any(stds([num])):
        if figures is None:
            figures = ''
        x = '{0:.{1:}uf}'.format(num, figures).replace('/', '')
    else:
        x = '{0:.{1:}f}'.format(num, figures)

    return r'\SI{{{}{}}}{{{}}}'.format(x, exp, unit)

def write(filename, content):
    f = codecs.open(filename, "w", "utf-8")
    if type(content) == uncertainties.core.Variable:
        content = "\num{" + str(x.n) + " +- " + str(x.s) + "}"
        f.write(content)
        if not content.endswith('\n'):
            f.write('\n')
        f.close()
    else:
        f.write(content)
        if not content.endswith('\n'):
            f.write('\n')
        f.close()

    #
    # with open(filename, 'w') as f:
    #     f.write(content)
    #     if not content.endswith('\n'):
    #         f.write('\n')


def make_full_table(caption,label,source_table, stacking=np.array([]), units=None):
    # Vorgeplänkel
    Output = """\\begin{table}
    \\centering
    \\caption{""" + caption + """}
    \\label{""" + label + """}
    \\sisetup{parse-numbers=false}
    \\begin{tabular}{\n"""

    # Kerngeschäft : source_table einlesen und verarbeiten, dh. Vor und Nachkommastellen rausfinden
    counter_columns = 0
    counter_lines = 0
    with open(source_table, 'r') as f:
        Text = f.read()
        for buchstabe in Text:
            if (buchstabe == '&'):
                counter_columns += 1
            elif (buchstabe == '\\'):
                counter_lines += 1

    NumberOfLines = counter_lines/2
    NumberOfColumns = counter_columns/counter_lines*2+1
    counter_digits_preDot = np.zeros((int(NumberOfLines), int(NumberOfColumns)), dtype=np.int)
    counter_digits_postDot = np.zeros((int(NumberOfLines), int(NumberOfColumns)), dtype=np.int)
    dot_reached = False
    counter_columns = 0
    counter_lines = 0
    with open(source_table, 'r') as f:
        Text = f.read()
    # 'Vor und Nachkommastellen rausfinden' beginnt hier
        for buchstabe in Text:
            if (buchstabe == '&'):
                counter_columns += 1
                dot_reached = False
            elif (buchstabe == '.'):
                dot_reached = True
            elif (buchstabe == '\\'):
                counter_lines += 1
                counter_columns = counter_columns % (NumberOfColumns-1)
                dot_reached = False
            elif (buchstabe != ' ') & (buchstabe != '\n'):
                if (counter_lines/2 <= (NumberOfLines-1)):
                    if dot_reached == False:
                        counter_digits_preDot[int(counter_lines/2)][int(counter_columns)] += 1
                    else:
                        counter_digits_postDot[int(counter_lines/2)][int(counter_columns)] += 1
    # jetzt ermittle maximale Anzahl an Stellen und speichere sie in MaxDigitsPreDot und MaxDigitsPostDot
    MaxDigitsPreDot = []
    counter_digits_preDot_np = np.array(counter_digits_preDot)
    for x in counter_digits_preDot_np.T:
        MaxDigitsPreDot.append(max(x))
    MaxDigitsPostDot = []
    counter_digits_postDot_np = np.array(counter_digits_postDot)
    for x in counter_digits_postDot_np.T:
        MaxDigitsPostDot.append(max(x))
    # --------------------Ende der Stellensuche

    # Die Liste stacking in ein angepasstes Array umwandeln mit den tatsächlich betroffenen Spalten
    stacking_list = np.array(stacking)
    i = 0
    for x in stacking_list:
        stacking_list[i] += i
        i += 1

    # Schreiben der Tabellenformatierung
    if np.size(stacking) == 0:
        for digits_preDot, digits_postDot in zip(MaxDigitsPreDot, MaxDigitsPostDot):
            Output += '\tS[table-format=' + str(digits_preDot) + '.' + str(digits_postDot) +']\n'
    else:   # es wurden fehlerbehaftete Werte übergeben, daher muss +- zwischen die entsprechenden Spalten
        i = 0.0
        for digits_preDot, digits_postDot in zip(MaxDigitsPreDot, MaxDigitsPostDot):
            if i in stacking_list:
                Output += '\tS[table-format=' + str(digits_preDot) + '.' + str(digits_postDot) +']\n'
                Output += '\t@{${}\\pm{}$}\n'
            elif i-1 in stacking_list:
                Output += '\tS[table-format=' + str(digits_preDot) + '.' + str(digits_postDot) +', table-number-alignment = left]\n'      # wir wollen hier linksbündige Zahlen
            else:
                Output += '\tS[table-format=' + str(digits_preDot) + '.' + str(digits_postDot) +']\n'
            i += 1

    # Zwischengeplänkel
    Output += '\t}\n\t\\toprule\n\t'

    # Einheitenzeile
    i=0
    stacking_list = np.array(stacking)
    for Spaltenkopf in units:
        if i in stacking_list:
            Output += '\\multicolumn{2}{c}'
        Output += '{' + str(Spaltenkopf) + '}\t\t'
        i += 1
        if i == np.size(units):
            Output += '\\\\ \n\t'
        elif i % 2 == 0:
            Output += '& \n\t'
        else:
            Output += '& '

    # Schlussgeplänkel
    Output += """\\midrule
    \\input{""" + source_table + """}
    \\bottomrule
    \\end{tabular}
    \\end{table}"""
    return Output
