import FD as fd

m = fd.Model({"x":range(1,10),"t":range(0,10)},"t",("x",))

f = fd.Field(m,"f",n_time_ders=1)
f.set_IC("x")
d1 = fd.Stencil([-1,0],1)

d1.der(f)