# Description

This is a library for setting up basic time-dependent finite difference problems. It handles the array manipulation and gives strict errors when a model isn't being set up correctly. However, it does not check for instability, nor does it check whether the model is actually physically accurate.

I see this mainly as an educational tool, although if more features are added it might be helpful for some research.


# Installation

```
pip install finite_difference
```

# Contributing
Any contribution or feedback is very welcome -- feel free to create an issue on github. In particular, if you are getting unexpected behavior or unclear errors, or if there's something you would like to see added, I would want to hear.

# Features
The library can support:
- Regular grid domains with arbitrary number of spatial dimensions
- Periodic or fixed boundary conditions
- Derivative approximations of arbitrary order and arbitrary sampled coordinates
- Time-dependent scalar fields with values at cells or edges between cells

# Example Use

Shown is all the code required for setting up a thermal diffusion model. For more examples, see [the Google Colab](https://colab.research.google.com/drive/1RL2nIeBTFvzbeLya2Qv0NR_kOcZW_Tr9#scrollTo=StZOQhW4wIzp).

1. Create a single Model object for the domain, with a time dimension and space dimension(s).

```python
import finite_difference as fd
m = fd.Model({"x": range(1,100,4), "t": range(1,10)}, time_axis = "t")
```

2. Create fields representing a property that changes over time as a scalar field. 

```python

T = fd.Field(m, "Temperature", n_time_ders = 1)
```

3. Create stencils for numerical approximations of spatial derivatives.

```python

diff_2 = fd.Stencil([-1,0,1], der_order = 2)
```


4. Apply boundary conditions and initial conditions.

```python

T.set_IC("1")
T.set_BC("0","x","start")
T.set_BC("0","x","end")
```

5. Run the simulation in a loop, updating the fields each iteration.

```python

k = 2 # thermal conductivity

m.check_IC() # not required, but recommended: checks if all necessary initial conditions have been set up

while not m.finished: # checks if it has reached the final timestep

    # implements the equations: dT/dt = k * d^2T/dx^2
    Tp = k*diff_2.der(T.prev)
    T.dot.assign_update(Tp)
    T.time_integrate_update()

    m.increment_time() # increment the time step
```

6. Once the run is complete, an interactive visual can be created showing the fields over time. Alternatively, you can get the values of the fields across all time as numpy arrays with the field's data property.


```python
m.interact() # creates an interactive visual in a jupyter notebook

# get numpy array of the temperature:
T.data      
```
