from FD import *

d = Domain({"x":range(0,5),"t":range(0,3)},time_axis="t")

cell = Field(d)
cell2 = Field(d)
edge = Field(d,edge_axes="x")
edge2 = Field(d,edge_axes="x")


d1 = Kernel([-1/2,5/2],1)



#cell.set_expression("x",{"t":0})

edge.set_expression("x",{"t":0})


cell.update_der(edge,d1,"x")

pass
