from sympy import Symbol, solve, sqrt, preorder_traversal, simplify, Eq, solveset, sin


x = Symbol('x')
exp = sin(x) + 2 - 14* x
print(solveset(exp, x))


"""def skobochki(exp):
    k = 0
    oppotunity_to_break = False
    for index, symbol in enumerate(exp):
        if not oppotunity_to_break:
            if symbol == '(':
                k += 1
                oppotunity_to_break = True
        else:
            if symbol == '(':
                k += 1
            if symbol == ')':
                k -= 1
            if k == 0:
                return exp[:index + 1]


def odz_znamenatel(exp):
    if exp.strip()[0] == '(':
        return skobochki(exp)
    elif exp.strip()[:4] == 'sqrt':
        return skobochki(exp)
    else:

        if '+' in exp:
            if '-' in exp:
                return exp[:min(exp.index('+'), exp.index('-'))]
            else:
                return exp[:exp.index('+')]
        if '-' in exp:
            if '+' not in exp:
                return exp[:exp.index('-')]
        return exp



def odz(exp):
    mas_odz_sqrt_exp = []
    mas_odz_znam_exp = []
    for arg in preorder_traversal(exp):
        if str(arg)[:4] == 'sqrt':
            new_exp = skobochki(str(arg))
            mas_odz_sqrt_exp.append(new_exp)
        if '/' in str(arg):
            new_exp = str(arg)[str(arg).index('/') + 1:]
            new_exp = odz_znamenatel(new_exp)
            mas_odz_znam_exp.append(new_exp)
    mas_odz_sqrt_exp = list(set(list(map(lambda x: x.strip(), mas_odz_sqrt_exp))))
    mas_odz_znam_exp = list(set(list(map(lambda x: x.strip(), mas_odz_znam_exp))))
    ODZ = [mas_odz_sqrt_exp, mas_odz_znam_exp]
    return ODZ





itog_odz = odz(exp)

print(itog_odz)
odz_sq = []
for i in itog_odz[0]:
    odz_sq.append(solve(i + '>= 0'))

odz_z = []
for i in itog_odz[1]:
    if 'x' in i:
        mas = []
        try:
            mas.append(solve(i + '< 0'))
        except:
            pass
        try:
            mas.append(solve(i + '> 0'))
        except:
            pass
        odz_z.append(mas)

stroka = ''
for index, el in enumerate(odz_sq):
    if index is not len(odz_sq) - 1:
        if str(el)[0] != '(':
            stroka += '(' + str(el) + ') & '
        else:
            stroka += '(' + str(el) + ') & '
    else:
        if str(el).strip()[0] != '(':
            stroka += '(' + str(el) + ')'
        else:
            stroka += '(' + str(el) + ')'


vnesh_stroka = ''
for index, el in enumerate(odz_z):
    vnutr_stroka = ''
    for index1, el1 in enumerate(el):
        if index1 is not len(el) - 1:
            if str(el1)[0] != '(':
                vnutr_stroka += '(' + str(el1) + ') | '
            else:
                vnutr_stroka += '(' + str(el1) + ') | '
        else:
            if str(el1)[0] != '(':
                vnutr_stroka += '(' + str(el1) + ')'
            else:
                vnutr_stroka += '(' + str(el1) + ')'

    if index is not len(odz_z) - 1:
        vnesh_stroka += str(vnutr_stroka) + ' & '
    else:
        vnesh_stroka += str(vnutr_stroka)


x = Symbol('x')

if stroka == '':
    if vnesh_stroka == '':
        pass
    else:
        new = '(' + vnesh_stroka + ')'
        print('ОДЗ:', new)

if vnesh_stroka == '':
    if stroka == '':
        pass
    else:
        new = '(' + stroka + ')'
        print('ОДЗ:', new)
if vnesh_stroka != '' and stroka != '':
    new = '(' + vnesh_stroka + ')' + ' & ' + '(' + stroka + ')'
    print('ОДЗ:', new)"""












