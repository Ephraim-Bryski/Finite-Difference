from FD import *
import numpy as np

m = Model(
    {"x":np.linspace(-5,5,100),
     #"y":np.linspace(-5,5,100),
     "t":range(0,2000)},
     periodic=["x"],
     time_axis="t"
)

# TODO should be some property set_IC sets to true --> if it's the first timestep and it IC haven't been applied and the field has a time derivative, throw an error
# wait actually, it's okay cause you'll get  

u = Field(m,edge_axes="x",n_time_ders=1)

eta = Field(m,n_time_ders=1)


#u.set_BC("0","x","start")
#u.set_BC("0","x","end")

#u.set_BC("0","y","start")
#u.set_BC("0","y","end")


eta.set_IC("exp(-x**2)")
u.set_IC("0")

#dudx = Field(m)
#dudy = Field(m)

c_e = Stencil([-1/2,1/2],1,axis_type="cell",der_axis_type="edge")
e_c = Stencil([-1,1],1,axis_type="edge",der_axis_type="edge")



cx = .1
cy = .1

g = 1
h = 1

# TODO add assertion 

while not m.finished:
    detadx = c_e.der(eta.prev,"x") # required to write u.prev
    
    # TODO assign_update doesn't give an error if the FieldInstant edges vs. cells is different than the field (might have checked size instead, but that doesn't work for PBC)
    u.dot.assign_update(g*detadx)

    u.time_integrate_update()

    dudx = e_c.der(u.new,"x")

    eta.dot.assign_update(h*dudx)


    eta.time_integrate_update()

    m.increment_time()
pass
