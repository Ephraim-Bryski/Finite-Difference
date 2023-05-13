This is a library for setting up basic time-dependent finite difference problems. It handles the array manipulation and gives strict errors when a model isn't being set up correctly. However, it does not check for instability, nor does it check whether the model is actually physically accurate -- that's still up to the user!

Procedure for performing a simulation:

1. The user creates a single Model object for the domain. The domain must have a time dimension and can have any number of spatial dimensions. Any of these spatial dimensions can be periodic.
2. The user creates Field objects representing a property that changes over time as a scalar field. For each spatial dimension, the values of the field can be at the cells or the edges between the cells. For example, for thermal diffusion, temperature would be at the cells, while temperature flux would be at the edges between the cells.
3. The user creates Stencil objects for numerical approximations of spatial derivatives. The coordinates to sample from and the derivative order is required to construct a stencil. For example, coordinates [-1,0,1] and order 2, would construct a stencil with values [1,-2,1] at the coordinates (approximation of a second derivative).
4. Once the user creates the needed Model, Fields, and Stencils, they must then run the simulation in a loop, updating the fields and their time derivatives using derivatives of fields and arithmetic operations between them. For efficiency, operations can only be done on fields at the current time.
5. Once the run is complete, an interactive visual can be created showing the fields over time. Alternatively, the user can get the values of the fields across all time as numpy arrays with the field's data property.

I see this as mainly an educational tool, although if more features are added (e.g. boundary value problems and vector fields), maybe it could be helpful for some research.

Any contribution or feedback is very welcome -- feel free to create an issue on github or email me at ebryski1@gmail.com.