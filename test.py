from FD import *
import numpy as np


d = Domain({"a":range(0,10),"b":range(0,10)})

#print(d.axes_lengths)


f = Field(d)
#f.set("a+1",location={"b":1})
f.set("exp(a**2+b**2)")



f2 = Field(d)
f2.set("1",location={"a":1})

fnew = f+f2
print(f.data)
print(f2.data)
print(fnew.data)