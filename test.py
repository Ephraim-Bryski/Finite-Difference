from FD import *
import numpy as np
import matplotlib.pyplot as plt


x_range = np.linspace(-5,5,100)

d = Domain({
    "x":x_range,
    "y":x_range
    ,"t":range(0,100)}
    ,time="t"
    ,periodic=["x","y"])


dt = d.dt
dx = dict(zip(d.axes_names,d.axes_step_size))["x"]


CFL = 0.2

k_coeff = CFL*dx**2/dt


print(CFL)
     
f = Field(d)
f.set_expression("exp(-x**2-y**2)",location={"t":0})
# f.set_expression("0",location={"x":0})
#f.set_expression("0",location={"x":-1})
#f.set_expression("0",location={"y":0})
#f.set_expression("0",location={"y":-1})


fxx = Field(d)
fyy = Field(d)

kx = Kernel([1,-2,1],center_idx=1,der_order=2,axis="x",domain=d)
ky = Kernel([1,-2,1],center_idx=1,der_order=2,axis="y",domain=d)


for i in range(d.n_time_steps-1):

    fxx.set_der(f,kernel=kx)
    fyy.set_der(f,kernel=ky)
    ft = k_coeff*(fxx+fyy)

    f.time_step(ft)

    d.increment_time()
pass




# define the kernel in the field? state that this field is the derivative of this other field with the following kernel
# seems more user friendly than initializing this abstract kernel object
# otherwise you would have to construct a kernel with the center at 1/2 AND specify the derivative is at the edges
# then could just run fp.take_diff()?

#fp = f.assign_derivative([-1,1],center_idx=1,dimension="x") # now no need to specify domain

# then it constructs a kernel and sets properties of fp with f and the kernel
