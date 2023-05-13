This is a library for setting up basic time-dependent finite difference problems. It handles the array manipulation and gives strict errors when a model isn't being set up correctly. However, it does not check for instability, nor does it check whether the model is actually physically accurate -- that's still up to the user!

Procedure for performing a simulation, the sample code showing the creation of a model for thermal diffusion:

0. Import the module.

```python
# TODO need to figure out what the import will look like
```

1. Create a single Model object for the domain. The domain must have a time dimension and can have any number of spatial dimensions. Any of these spatial dimensions can be periodic, but in this case won't be.

```python
m = FD.Model({"x": range(1,100,4), "t": range(1,10)}, time_axis = "t")
```

2. Create Field objects representing a property that changes over time as a scalar field. For each spatial dimension, the values of the field can be at the cells or the edges between the cells. For example, for thermal diffusion, temperature would be at the cells, while temperature flux would be at the edges between the cells.

```python
T = FD.Field(m, "Temperature", n_time_ders = 1)
Tflux = FD.Field(m, "Temperature Flux", n_time_ders = 0, edge_axes = "x")

```

3. The user creates Stencil objects for numerical approximations of spatial derivatives. The coordinates to sample from and the derivative order is required to construct a stencil.

```python
cell_to_edge = FD.Stencil([-1/2,1/2],der_order=1,axis_type="cell",der_axis_type="edge")
edge_to_cell = FD.Stencil([-1/2,1/2],der_order=1,axis_type="edge",der_axis_type="cell")
```

The resulting equations are printed, the same for both stencils:

Finite approximation: f' = [-f(x-0.5h) + f(x+0.5h)] / [h^1]

4. Boundary conditions (for nonperiodic axes) and initial conditions (for fields with time derivatives) are applied. In this case, the temperature is fixed on one end and the flux is fixed on the other.

```python
T.set_IC("1")
T.set_BC("0","x","start")
Tflux.set_BC("0","x","end")
```

Note that the boundary condition overrides the initial condition for temperature; the user is warned about this.

5. Once the user creates the needed Model, Fields, and Stencils, they must then run the simulation in a loop, updating the fields and their time derivatives using derivatives of fields and arithmetic operations between them. For efficiency, operations can only be done on fields at the current time.

```python

k = 2 # thermal conductivity

# optional check for stability, note that this criteria is only for this particular problem:
CFL = k*dt/dx**2
print(f"CFL: {round(CFL,3)}, must be under 0.5 for stability\n")

m.check_IC() # not required, but recommended: check's if all necessary initial conditions have been set up

while not m.finished:

    dTdx = cell_to_edge.der(T.prev,"x")

    Tflux.assign_update(k * dTdx)

    Tp = edge_to_cell.der(Tflux.new,"x")

    T.dot.assign_update(Tp)

    T.time_integrate_update()

    m.increment_time()
```

6. Once the run is complete, an interactive visual can be created showing the fields over time. Alternatively, the user can get the values of the fields across all time as numpy arrays with the field's data property.


```python
m.interact() # creates an interactive visual in a jupyter notebook

# get numpy arrays of the temperature and temperature flux:
Tflux.data
T.data      
```

I see this as mainly an educational tool, although if more features are added (e.g. boundary value problems and vector fields), maybe it could be helpful for some research.

Any contribution or feedback is very welcome -- feel free to create an issue on github or email me at ebryski1@gmail.com.