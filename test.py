from FD import *




t = Field({"a":3,"b":4,"c":2},None)


mini = Field({"c":4,"b":2},4)

t[{"a":2}] = mini 

t[{"a":2,"b":3}] = Field({"c":2},999)
mini.show(rows=["b"],cols=["c"])
t.show(rows=["a","b"],cols=["c"])
some_slice = t[{"b":3}]
some_slice.show(rows=["a"],cols=["c"])
