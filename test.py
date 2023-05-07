from FD import *
import numpy as np
import line_profiler






x_range = np.linspace(0,10,50)

d = Domain({
    "x":x_range,
    "y":range(0,3),
    "t":range(0,500)}
    ,time_axis="t"
    ,periodic=["y","x"])


dt = d.dt
dx = dict(zip(d.axes_names,d.axes_step_size))["x"]


CFL = 0.3

k_coeff = CFL*dx**2/dt


     
f = Field(d)


f.set_expression("exp(-x**2)",location={"t":0})


ft = Field(d)
ft.set_expression("0",location={"t":0})



f.set_expression("0",location={"x":0})
f.set_expression("0",location={"x":-1})
#f.set_expression("0",location={"y":0})
#f.set_expression("0",location={"y":-1})


fxx = Field(d)
fyy = Field(d)

kx = Kernel([1,-2,1],center_idx=1,der_order=2,axis="x",domain=d)
ky = Kernel([1,-2,1],center_idx=1,der_order=2,axis="y",domain=d)


for i in range(d.n_time_steps-1):

    fxx.update_der(f,kernel=kx)
    fyy.update_der(f,kernel=ky)

    ftt = k_coeff*(fxx+fyy)

    if i%50==0:
        print(i)
    d.update_time((ftt,ft,f))
    #f.update_time_step(ft)

pass

print("HI")

plt.show()

