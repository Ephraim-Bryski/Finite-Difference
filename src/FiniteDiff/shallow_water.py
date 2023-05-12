from FD import *
import numpy as np

m = Model(
    {"x":np.linspace(-5,5,100),
     "y":np.linspace(-5,5,100),
     "t":range(0,200)},
     periodic=["y"],
     time_axis="t"
)


u = Field(m,"u",edge_axes="x",n_time_ders=1)
v = Field(m,"v",edge_axes="y",n_time_ders=1)

eta = Field(m,"eta",n_time_ders=1)


u.set_BC("0","x","start")
u.set_BC("0","x","end")

#u.set_BC("0","y","start")
#u.set_BC("0","y","end")


eta.set_IC("exp(-(x-2)**2-(y-2)**2)")
u.set_IC("0")
v.set_IC("0")

#dudx = Field(m)
#dudy = Field(m)

c_e = Stencil([-1/2,5/2],1,axis_type="cell",der_axis_type="edge")
e_c = Stencil([-1/2,1/2],1,axis_type="edge",der_axis_type="cell")


cx = .1
cy = .1

g = .1
h = 1

while not m.finished:
    detadx = c_e.der(eta.prev,"x") # required to write u.prev

    detady = c_e.der(eta.prev,"y")

    u.dot.assign_update(g*detadx)

    v.dot.assign_update(g*detady)

    u.time_integrate_update()
    v.time_integrate_update()

    dudx = e_c.der(u.new,"x")
    dvdy = e_c.der(v.new,"y")

    eta.dot.assign_update(h*dudx+h*dvdy)


    eta.time_integrate_update()

    m.increment_time()
pass


