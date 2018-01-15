''' Integer to English words
'''

def numberToWords(num):
    if 0 == num: return 'Zero'

    kilo_units = ['', 'Thousand', 'Million', 'Billion']
    cent_units = ['', '', 'Twenty', 'Thirty', 'Forty', 'Fifty', 
                  'Sixty', 'Seventy', 'Eighty', 'Ninety']
    units = ['', 'One', 'Two', 'Three', 'Four', 'Five',
             'Six', 'Seven', 'Eight', 'Nine', 'Ten', 
             'Eleven', 'Twelve', 'Thirteen', 'Fourteen',
             'Fifteen', 'Sixteen', 'Seventeen', 'Eighteen', 'Nineteen']
    
    def parse_range_kilo(n, and_flag=False):
        if 0 == n: return ''
        res = []
        if n >= 100:
            res += [units[n // 100] + ' Hundred']
            n = n % 100
        if n >= 20:
            curr = cent_units[n // 10]
            if and_flag:
                res += ['and', curr]
                and_flag = False
            else:
                res += [curr]
            n = n % 10
        if n > 0:
            curr = units[n]
            if and_flag:
                res += ['and', curr]
            else:
                res += [curr]

        return ' '.join(res)
    
    i = 0
    res = []
    and_flag = num >= 100
    while num > 0:
        a = num % 1000
        num = num // 1000
        if 0 != a:            
            _kilo_str = parse_range_kilo(a, and_flag)
            and_flag = False
            curr = _kilo_str + ('' if 0 == i else ' ' + kilo_units[i])
            res += [curr]
        i += 1

    return ' '.join(res[::-1])


