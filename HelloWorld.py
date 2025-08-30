import numpy as np
data = [np.random.standard_normal() for i in range(7)]
print(data)

def isiterable(obj):
     try:
         iter(obj)
         print("True")
     except TypeError: # not iterable
         print("False")
     
isiterable("a string")
isiterable([1,2,3]) 
isiterable(5)
isiterable({'a':1, 'b':2})
isiterable(data)