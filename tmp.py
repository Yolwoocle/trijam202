import sortedcontainers
import time

class Temp:
    def __init__(self, val):
        self.val = val
    
    def __str__(self) -> str:
        return str(self.val)
    
    def __le__(self, other:'Temp'):
        return self.val<=other.val

    def __lt__(self, other:'Temp'):
        return self.val<other.val

L = []
a = Temp(5)
b = Temp(6)
L.append(a)
L.append(b)

# print(L)
for x in L: print(x, end=None)
print("")

time.sleep(1)
b.val=2
L.sort()
time.sleep(1)

for x in L: print(x, end=None)
print("")
