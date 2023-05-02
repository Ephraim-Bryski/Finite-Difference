from FD import *
import numpy as np





d = Domain({"a":range(-5,5),"b":range(-5,5)},periodic=["b"])

f = Field(d)

f.set("a**2*b**2")


k1 = Kernel([1,-2,1],1,"a",d)
k2 = Kernel([1,-2,1],1,"b",d)


f.diff(k1).show(rows=["a"])
f.diff(k1).diff(k1).show(rows=["a"])

pass

# custom type?? s




def kernel(der_order,approx_order,type,step):

    possible_types = ["forward","backward","central"]
 
    assert type in possible_types, f"types must be one of {possible_types}"

    combination = (der_order,approx_order,type)

    if combination == (1,1,"forward"):
        kernel = {0:-1,1:1}
    elif combination == (1,2,"central"):
        kernel = {-1:-0.5,0:0,1:0.5}
    elif combination == (1,1,"backward"):
        kernel =  {-1:-1,0:1}
    elif combination == (2,1,"central"):
        kernel = {-2:-1,0:0,1:1}
    else:
        raise Exception("not valid or haven't implemented")

    kernel = {kernel[k]/step**der_order for k in kernel}
    return kernel


kernel(1,2,"central",1)


pass