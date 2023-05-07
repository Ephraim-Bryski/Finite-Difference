from FD import *
import numpy as np
import line_profiler

@profile
def hi():
    print("HI")

x_range = np.linspace(-5,5,100)

d = Domain({
    "x":x_range,
    "y":x_range
    ,"t":range(0,2)}
    ,time_axis="t"
    ,periodic=["x","y"])


dt = d.dt
dx = dict(zip(d.axes_names,d.axes_step_size))["x"]


CFL = 0.2

k_coeff = CFL*dx**2/dt


print(CFL)
     
f = Field(d)


f.set_expression("exp(-x**2-y**2)",location={"t":0})

f.imshow({"t":0})
plt.figure()

f.plot({"t":0,"x":0})

plt.show(block=False)


# f.set_expression("0",location={"x":0})
#f.set_expression("0",location={"x":-1})
#f.set_expression("0",location={"y":0})
#f.set_expression("0",location={"y":-1})


fxx = Field(d)
fyy = Field(d)

kx = Kernel([1,-2,1],center_idx=1,der_order=2,axis="x",domain=d)
ky = Kernel([1,-2,1],center_idx=1,der_order=2,axis="y",domain=d)


for i in range(d.n_time_steps-1):

    fxx.update_der(f,kernel=kx)
    fyy.update_der(f,kernel=ky)

    ft = k_coeff*(fxx+fyy)

    f.update_time_step(ft)

    d.increment_time()
pass

print("HI")

plt.show()

