from typing import List


def hello(name: str = None) -> str:
    return ("Hello, " + name + "!") if name else "Hello!"


def int_to_roman(num: int) -> str:
    dic = {1000: "M",
           900: "CM",
           500: "D",
           400: "CD",
           100: "C",
           90: "XC",
           50: "L",
           40: "XL",
           10: "X",
           9: "IX",
           5: "V",
           4: "IV",
           1: "I"}
    s = ""
    for number, value in dic.items():
        while num >= number:
            s = s + value
            num -= number
    return s


def longest_common_prefix(strs_input: List[str]) -> str:
    if not strs_input:
        return ''
    pref = strs_input[0].strip()
    for i in strs_input:
        j = 0
        k = i.strip()
        while j < len(k) and j < len(pref) and pref[j] == k[j]:
            j += 1
        pref = pref[:j]
    return pref


def primes() -> int:
    i = 2
    while True:
        flag = True
        for j in range(2, round(i**(1/2)) + 1):
            if i % j == 0:
                flag = False
                break
        if flag:
            yield i
        i += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    @property
    def balance(self):
        if self.balance_limit != None and (self.balance_limit - 1 < 0):                
            print("Balance check limits exceeded.")
            raise ValueError
        else:
            if self.balance_limit != None:
                self.balance_limit -= 1
            return self.total_sum
    
    def __call__(self, sum_spent: int):
        if (self.total_sum - sum_spent) < 0:
            print("Not enough money to spend ",sum_spent," dollars.")
            raise ValueError
        else: 
            self.total_sum -= sum_spent
            print("You spent ",sum_spent," dollars.")

    def __repr__(self):
        return f"To learn the balance call balance."

    def put(self, sum_put: int):
        self.total_sum += sum_put
        print("You put ",sum_put," dollars.")

    def __add__ (self, b: 'BankCard'):
        self.total_sum += b.total_sum
        self.balance_limit = max(self.balance_limit, b.balance_limit)
        return self