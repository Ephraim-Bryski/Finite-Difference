This is a library for setting up basic time-dependent finite difference problems. It handles the array manipulation and gives strict errors when a model isn't being set up correctly. However, it does not check for instability, nor does it check whether the model is actually physically accurate -- that's still up to the user!


I see this as mainly an educational tool, although if more features are added (e.g. boundary value problems and vector fields), maybe it could be helpful for some research.

Any contribution or feedback is very welcome -- feel free to create an issue on github or email me at ebryski1@gmail.com.



Procedure for performing a simulation, the sample code showing the creation of a model for thermal diffusion:

For more examples, see [the Google Colab](https://colab.research.google.com/drive/1RL2nIeBTFvzbeLya2Qv0NR_kOcZW_Tr9#scrollTo=StZOQhW4wIzp)

0. Import the module.

```python
# TODO need to figure out what the import will look like
```

1. Create a single Model object for the domain, with a time dimension and space dimension(s).

```python

m = FD.Model({"x": range(1,100,4), "t": range(1,10)}, time_axis = "t")
```

2. Create fields representing a property that changes over time as a scalar field. 

```python

T = FD.Field(m, "Temperature", n_time_ders = 1)
```

3. Create stencils for numerical approximations of spatial derivatives.

```python

diff_2 = FD.Stencil([-1,0,1],der_order=2)
```


4. Boundary conditions and initial conditions are applied.

```python

T.set_IC("1")
T.set_BC("0","x","start")
T.set_BC("0","x","end")
```

5. Run the simulation in a loop, updating the fields each iteration.

```python

k = 2 # thermal conductivity

m.check_IC() # not required, but recommended: check's if all necessary initial conditions have been set up

while not m.finished: # checks if it his reached the final timestep

    # implements the equations: dT/dt = k * d^2T/dx^2
    Tp = k*diff_2.der(T.prev)
    T.dot.assign_update(Tp)
    T.time_integrate_update()

    m.increment_time() # increment the time step
```

6. Once the run is complete, an interactive visual can be created showing the fields over time. Alternatively, the user can get the values of the fields across all time as numpy arrays with the field's data property.


```python
m.interact() # creates an interactive visual in a jupyter notebook

# get numpy arrays of the temperature:
T.data      
```