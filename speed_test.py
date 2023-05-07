
from FD import *
import cProfile
import pstats
import os


from FD import *
import numpy as np


def run_model():

    x_range = np.linspace(-5,5,100)

    d = Domain({
        "x":x_range,
        "y":x_range
        ,"t":range(0,5)}
        ,time_axis="t"
        ,periodic=["x","y"]
        ,check_bc = True)


    dt = d.dt
    dx = dict(zip(d.axes_names,d.axes_step_size))["x"]


    CFL = 0.4

    k_coeff = CFL*dx**2/dt


        
    f = Field(d)


    f.set_expression("exp(-x**2-y**2)",location={"t":0})


    bc = False
    if bc:
        f.set_expression("0",location={"x":0})
        f.set_expression("0",location={"x":-1})
        f.set_expression("0",location={"y":0})
        f.set_expression("0",location={"y":-1})


    fxx = Field(d)
    fyy = Field(d)

    kx = Kernel([1,-2,1],center_idx=1,der_order=2,axis="x",domain=d)
    ky = Kernel([1,-2,1],center_idx=1,der_order=2,axis="y",domain=d)

    ft = Field(d)

    for _ in range(d.n_time_steps-1):

        fxx.update_der(f,kernel=kx)
        fyy.update_der(f,kernel=ky)

        ft.update_values(k_coeff*(fxx.now+fyy.now))
        
        f.update_time_step(ft)

        d.increment_time()

with cProfile.Profile() as profile:
    run_model()

results = pstats.Stats(profile)

f_name = "boop.prof"
results.dump_stats(f_name)
os.system(f"snakeviz {f_name}")




