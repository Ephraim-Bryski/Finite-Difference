from FD import *
import numpy as np
import matplotlib.pyplot as plt

n_x = 100 # TODO obviously you have to be allowed to say you want the end to be fixed, temporary solution


d = Domain({
    "x":np.linspace(0,1,n_x)
    ,"t":range(0,50)}
    ,time="t"
    ,periodic=[])
     
f = Field(d)
f.set_expression("1",location={"t":0})

fxx = Field(d)
f.set_expression("0",location={"x":0})
f.set_expression("0",location={"x":-1})


k = Kernel([1,-2,1],center_idx=1,axis="x",domain=d)

for i in range(d.n_time_steps-1):
    fxx.set_der(f,kernel=k)
    ft = 0.2*fxx

    f.time_step(ft)

    d.increment_time()
pass




# define the kernel in the field? state that this field is the derivative of this other field with the following kernel
# seems more user friendly than initializing this abstract kernel object
# otherwise you would have to construct a kernel with the center at 1/2 AND specify the derivative is at the edges
# then could just run fp.take_diff()?

#fp = f.assign_derivative([-1,1],center_idx=1,dimension="x") # now no need to specify domain

# then it constructs a kernel and sets properties of fp with f and the kernel
