import FD as FD

import imp
imp.reload(FD)

m = FD.Model({"x":[1,4],"y":[1,5],"t":range(0,10,2)},time_axis="t")

f = FD.Field(m,name="hi",n_time_ders=1)

f2 = FD.Field(m,name="yo")

f.set_IC("x")
f.set_BC("0","x","end")

#f.set_IC("x")

m.interact()
pass