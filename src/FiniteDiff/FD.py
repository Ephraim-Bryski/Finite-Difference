import numpy as np
import matplotlib.pyplot as plt
import operator
import numexpr
import itertools
import warnings
import numbers
import math
import ipywidgets.widgets as widgets
import re



class Model:

    def __init__(self, axes: dict, time_axis: str, periodic: tuple = ()):

        """
        creates the domain which will be used for the simulation

        axes: the keys are the name of the axis, and the values are an evenly \
            spaced array of values for the given axis
        time_axis: the axis name that represents time
        periodic: tuple of axes names that are periodic
        """

        # check types:
        assert type(axes) == dict, "axes must be a dictionary of axes and corresponding values"
        assert type(time_axis) == str, "time_axis must be a string"

        if type(periodic) == list:
            periodic = tuple(periodic)
        assert type(periodic) == tuple, "periodic must be a list or tuple"

        # check axes:
        def array_error_msg(axis_name, vals):

            if type(axis_name) != str:
                return "axes keys must be strings, the name of the axis"

            try:
                iter(vals)
            except TypeError:
                return "axes must be iterable"

            if not np.all([isinstance(val, numbers.Number) for val in vals]):
                return "all values of axis must be numbers"

            if len(vals) < 2:
                return "must have more than one value in axis"

            diffs = np.diff(vals)
            tolerance = 10**-9  # numpy linspace isn't perfectly spaced
            if not np.all([abs(diffs[0]-diff) < tolerance for diff in diffs]):
                return "all values of axes must be evenly spaced"
            
            if diffs[0]<tolerance:
                return "axis values cannot be repeated"

        for axis_name, axis_vals in axes.items():

            msg = array_error_msg(axis_name, axis_vals)
            if msg is not None:
                raise Exception(f"error with axis {axis_name}: {msg}")

            axes[axis_name] = np.array(axis_vals)

        # check time axes:
        assert type(time_axis) == str, "time axis must be a string"
        assert time_axis in axes, f"time axis must be one of the axes names, \
{time_axis} is not"

        assert time_axis not in periodic, "time axis can't be periodic"

        # check periodic:
        non_axes_periodic = set(periodic)-set(axes.keys())
        assert non_axes_periodic == set(), f"periodic axes must be one of the axes names, \
{non_axes_periodic} is/are not"

        # add properties
        self.axes = axes
        self.periodic = periodic
        self.time_step = 0
        self.time_axis = time_axis
        self.fields = []
        self.insufficient_bc = False # this is set to true when data is getting lost, this way it doesn't warn over and over

    def check_IC(self):
        """
        checks if all fields with time derivatives have had their initial conditions set
        """
        for field in self.fields:
            if field.dot is not None and not field.IC_set:
                raise Exception(f"field {field.name} has not had its initial conditions set")
            
        print("All initial conditions set!\n")

    def increment_time(self):

        """
        must be called for each iteration of the simulation
        increments time step and resets all field's updated property to False
        """

        def clear_update(field):
            field.updated = False
            if field.dot is not None:
                clear_update(field.dot)

        for field in self.fields:
            if isinstance(field,ConstantField):
                continue
            assert field.updated, f"all fields must be updated in the iteration, {field.name} has not been"
            clear_update(field)

        self.time_step += 1

    def interact(self):

        """
        generates interactive visual of the fields over time
        line plot for model with one time axis
        imshow for model with two time axes
        """

        assert self.time_axis is not None, "need time axis to use interact"
        assert len(self.fields) != 0, "model has no fields"


        # reversed just so time derivatives come after their fields --> weird to have a time derivative to show up first
        interact_fields = list(self.fields)
        interact_fields.reverse()

        field_names = [field.name for field in interact_fields]

        field_dropdown = widgets.Dropdown(
            options=field_names, description='Field')

        time_values = self.axes[self.time_axis]

        min_time = time_values[0]
        dt = self.dt
        max_time = time_values[-1]

        n_plot_dims = len(self.axes)-1  # subtracting one due to time axis

        plt.close("all")
        _, ax = plt.subplots(figsize=(6, 4))

        space_axes = self.axes.copy()

        space_axes.pop(self.time_axis)

        if n_plot_dims > 2:
            raise Exception("for now can only visualize \
                            with 1 or 2 space dimensions")

        time_axis_idx = self.axes_names.index(self.time_axis)

        @widgets.interact(field_name=field_dropdown, t=(min_time, max_time, dt))
        def update(field_name=field_names[0], t=0):
            ax.clear()
            field_idx = field_names.index(field_name)

            field = interact_fields[field_idx]
            data = field.data

            # min and max across all data so that the color scale/ylim doesn't change moving through time
            max_val = np.nanmax(data)
            min_val = np.nanmin(data)

            # only relevant if everything is nan (but prevents error when setting the values)
            if np.isnan(max_val):
                max_val = 0.1
            if np.isnan(min_val):
                min_val = 0

            time_values = self.axes[self.time_axis].tolist()
            time_diffs = [abs(t-time_val) for time_val in time_values]
            time_idx = time_diffs.index(min(time_diffs))

            # accessing data directly instead of using __get_data from Field class
            slice_idxs = [slice(None)]*len(data.shape)
            slice_idxs[time_axis_idx] = time_idx

            data_slice = data[tuple(slice_idxs)]

            space_names = list(space_axes.keys())
            space_values = list(space_axes.values())

            x_values = space_values[0]
            ax.set_xlim(x_values[0], x_values[-1])
            ax.set_xlabel(space_names[0])

            if n_plot_dims == 1:
                ax.set_ylabel(field_name)
                ax.set_ylim(min_val, max_val)
                # very hacky, but nonperiodic edge axes are one longer in length, so I need to remove one to plot
                if len(data_slice)>len(x_values):
                    data_slice = data_slice[:-1]
                ax.plot(x_values, data_slice, color="blue")

            else:
                y_values = space_values[1]
                ax.set_ylabel(space_names[1])
                ax.set_ylim(y_values[0], y_values[-1])

                # this is also used (and explained) in the Field imshow method:
                im_data = np.flipud(np.transpose(data_slice))

                ax.imshow(im_data,
                          extent=[x_values[0], x_values[-1],
                                  y_values[0], y_values[-1]],
                          vmin=min_val,
                          vmax=max_val
                          )

    @property
    def finished(self):

        """
        true only when the simulation has finished running
        this means the timestep has reached the maximum number set for the mdoel
        """

        time_values = self.axes[self.time_axis]
        n_steps = len(time_values)
        if self.time_step >= n_steps:
            raise ValueError(
                "the current timestep reached the number of timestips -- this shouldn't happen")
        return self.time_step+1 == n_steps # +1 used since when setting a new value, it sets the i+1th time slice

    @property
    def dt(self):
        assert self.time_axis != None, "no time axis"
        return dict(zip(self.axes_names, self.axes_step_size))[self.time_axis]

    @property
    def n_time_steps(self):
        assert self.time_axis != None, "no time axis"
        return dict(zip(self.axes_names, self.axes_lengths))[self.time_axis]

    @property
    def axes_lengths(self):
        return [len(list(axis_range)) for axis_range in self.axes.values()]

    @property
    def axes_names(self):
        return list(self.axes.keys())

    @property
    def axes_values(self):
        return list(self.axes.values())

    @property
    def axes_step_size(self):
        # this assumes the axes are evenly spaced, which is checked in initialization
        return [axis_range[1]-axis_range[0] for axis_range in self.axes.values()]

class Field:

    def __init__(self, model: Model, name: str, edge_axes: tuple=(), n_time_ders: int=0):

        """
        Fields are scalar fields which are updated during the simulation

        model: a Model object which the field is part of, all fields should share the same model
        name: field name, used for labeling plots
        edge_axes: the axes along which the values of the field are at the edges, between cells
        n_time_ders: number of time derivatives of the field which are used for modelling,
        use the dot property to get the time derivative of a field
        """

        assert isinstance(model, Model), "model must be a Model object"
        assert type(name) == str, "field name must be a string"
        assert type(edge_axes) in [str, list, tuple], "edge axes must be a string, list or tuple"
        
        axes = model.axes_names
        assert set(edge_axes) - set(axes) == set(), \
            "edge_axes must be axes in the model"
        
        assert model.time_axis not in edge_axes, \
            "edge_axes cannot include the time axis"

        assert type(n_time_ders) == int and n_time_ders >= 0, \
            "n_time_ders must be an integer greater than equal to 0"
            
        edge_axes = tuple(edge_axes)

        self.name = name

        # recursively construct fields for time derivatives:
        if n_time_ders > 0:
            self.dot = Field(model, f"{self.name} dot", edge_axes, n_time_ders-1)
        else:
            self.dot = None

        model.fields.append(self)

        # all non-periodic edges axes are one greater in length (values on both end edges)
        axes_lengths = dict(zip(model.axes_names, model.axes_lengths))
        for axis in edge_axes:
            if axis not in model.periodic:
                axes_lengths[axis] += 1

        axes_lengths = tuple(axes_lengths.values())

        self.model = model
        self.edge_axes = edge_axes
        self.data = np.full(axes_lengths, np.nan)
        self.updated = False
        self.IC_set = False # set to True when set_IC method ran, so check_IC can check

    @property
    def new(self):

        """
        Returns the section of the field at the upcoming time
        Can only be used after the field has been updated
        Use the prev method before the field has been updated
        """

        assert self.updated, "field has not yet been updated, use prev property to get current values"
        current_time = self.model.time_step+1
        time_axis = self.model.time_axis
        return self.__get_data({time_axis: current_time})

    @property
    def prev(self):

        """
        Returns the section of the field at the current time
        Can only be used before the field has been updated
        Use the new method after the field has been updated
        """

        assert not self.updated, "field has already been updated, use new property to get values at the next timestep\n\
            alternatively, you may have forgot to call the increment_time method at the end of the iteration "
        assert self.dot is not None, "cannot access prev for a field without a time derivative since no initial conditions are set,\n\
instead first use the assign_update method and then use the new property"
        previous_time = self.model.time_step
        time_axis = self.model.time_axis
        return self.__get_data({time_axis: previous_time})

    def set_IC(self, expression:str):

        """
        sets the values of the field at t=0
        performed before the simulation iterations
        only can be performed on fields with time derivatives

        expression: the mathematical expression for the field, as a function of the spatial axes
        """

        assert self.dot != None, "field must have a time derivative to set initial conditions"
        time_axis = self.model.time_axis
        assert time_axis != None, "the model must have a time axis to set the initial conditions"
        self.__set_expression(expression, {time_axis: 0})
        self.IC_set = True # this is only used so the model can check if all ICs with time derivatives have been set before running 

    def set_BC(self, expression:str, axis: str, side: str):

        """
        sets the values of the field at boundaries of the domain

        expression: the mathematical expression for the field, as a function of the axes
        axis: the axis which the boundary condition is applied to
        side: "start" or "end", which side the boundary condition is applied to

        the expression cannot be a function of the axis
        """

        assert axis != self.model.time_axis, "cannot use BC to set time axis, use set_IC instead"
        assert axis in self.model.axes_names, "axis must be one of the axes names"
        assert axis not in self.model.periodic, "cannot set boundary conditions to periodic axis"

        side_types = ["start", "end"]
        assert side in side_types, f"side must be one of {side_types}"

        if side == "start":
            idx = 0
        elif side == "end":
            idx = -1

        self.__set_expression(expression, {axis: idx})

    def __set_expression(self, expression: str, location: dict={}):

        """
        sets the data to the value of the expression at the specified location
        used for boundary conditions and initial conditions
        """

        assert type(expression) == str, "expression must be a string"
        assert type(location) == dict

        def transpose(nested_list):
            transposed_list = [[row[i] for row in nested_list]
                               for i in range(len(nested_list[0]))]
            return transposed_list

        def extract_variables(expression):
            # Regular expression pattern for variable names
            pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
            math_terms = ["exp","abs","sin","cos","tan","arcsin","arccos","arctan","sinh","cosh","tanh"]
            variables = re.findall(pattern, expression)
            variables = [var for var in variables if var not in math_terms]
            return variables

        substitute_axes_names = []
        substitute_axes_values = []
        substitute_axes_lengths = []

        # combine all the values to substitute for in a list
        for i in range(len(self.model.axes_names)):

            axis_name = self.model.axes_names[i]
            axis_values = self.model.axes_values[i]
            axis_length = self.model.axes_lengths[i]

            # only evaluating along dimensions not specified in location
            if axis_name in location.keys():
                continue

            edge_axis = axis_name in self.edge_axes
            periodic = axis_name in self.model.periodic

            if edge_axis:
                # shift the values to the left by delta/2
                step_sizes = dict(
                    zip(self.model.axes_names, self.model.axes_step_size))
                dx = step_sizes[axis_name]

                axis_values = [val-dx/2 for val in axis_values]

            if edge_axis and not periodic:
                # for edge axis that aren't periodic, there's an extra value at the end (5 cells means 6 edges)
                # if it's periodic, it's the same amount since the two edges on the end are the same
                final_value = axis_values[-1]
                axis_values.append(final_value+dx)
                axis_length += 1

            substitute_axes_names.append(axis_name)
            substitute_axes_values.append(axis_values)
            substitute_axes_lengths.append(axis_length)

        # checking variables this way identifies possible errors before passed to numexpr
        variables = extract_variables(expression)
        unknown_vars = set(variables)-set(self.model.axes_names)
        if len(unknown_vars) != 0:
            raise Exception(f"variables {unknown_vars} are not axes names")
        unsubbed_vars = set(variables)-set(substitute_axes_names)
        if len(unsubbed_vars) != 0:
            raise Exception(f"variables {unsubbed_vars} are axes names along which the expression is being set")

        # constructs nested list of all combinations of values
        value_combs = transpose(
            [list(comb) for comb in itertools.product(*substitute_axes_values)])
        subs = dict(zip(substitute_axes_names, value_combs))

        try:
            data_flat = numexpr.evaluate(expression, subs)
        except:
            raise Exception("cannot parse expression")

        # if the expression is just a constant, numexpr just returns a single value instead of an array
        if data_flat.shape == ():
            n_subs = len(value_combs[0])
            data_flat = np.full((n_subs, 1), float(data_flat))

        data = np.reshape(data_flat, substitute_axes_lengths)

        field_slice = FieldInstant(self.model, self.edge_axes, self.name, data)

        # allow override allows you to (for example) override boundary conditions with initial conditions or vice versa
        self.__set_data(field_slice, location, allow_override=True)

    def assign_update(self, field_slice):

        """
        assigns the values of field_slice to a field at a current time
        field can only be updated once each iteration

        field_slice: a slice of a field at the current time,\
            returned from now and prev methods, math operations on these objects,\
            or derivatives on these objects
        """

        assert not self.updated, "field already updated, you may have forgot to call increment_time on the model"
        assert self.dot == None, "cannot use assign_update on field with time derivative, use time_integrate_update instead"
        assert isinstance(field_slice, FieldInstant), "invalid argument, instead use the 'prev' or 'new' properties of a field (or operations on them)"

        time_axis = self.model.time_axis
        time = self.model.time_step+1


        self.__set_data(field_slice, {time_axis: time}, allow_override=False)
        self.updated = True

    def time_integrate_update(self):

        """
        updates a field using Euler integration of it's time derivative
        field can only be updated once each iteration

        field_slice: a slice of a field at the current time,\
            returned from now and prev methods, math operations on these objects,\
            or derivatives on these objects
        """

        assert not self.updated, "field already updated"
        assert self.dot != None, "field needs to have time derivative to perform time_integrate"
        assert self.dot.updated, "time derivative needs to be updated first, can be accessed with 'dot' property"

        dt = self.model.dt
        new_slice = self.prev + self.dot.new*dt

        time_axis = self.model.time_axis
        time = self.model.time_step+1  # +1 since it's updating the following value

        prev_data = self.prev.data

        self.__set_data(new_slice, {time_axis: time}, allow_override=False)
        self.updated = True

        new_data = self.new.data

        prev_nan_count = np.isnan(prev_data).sum()
        new_nan_count = np.isnan(new_data).sum()

        # im not 100% sure if this will always work correctly, 
        # but if there's more unknowns than there were before,
        # and it's not the first timestep,
        # that's an indication the unknowns are spreading
        if time>1 and new_nan_count > prev_nan_count and not self.model.insufficient_bc:
            print("Warning: fields seem to be losing data over time\n\
This is possibly due to insufficent boundary conditions\n")
            self.model.insufficient_bc = True # so it doesn't give the warning over and over again


    def __set_data(self, field_slice, location, allow_override):

        """
        combines self's data at the location with the input data (merging unknown and known values)
        then sets self's data to merged data
        """

        assert isinstance(field_slice, FieldInstant), "must input field slice"

        field_edge_axes = set(self.edge_axes)-set(location.keys())
        slice_edge_axes = set(field_slice.edge_axes)-set(location.keys())

        assert field_edge_axes == slice_edge_axes, "field edge axes and field slice edge axes must match"
        # check that edge_axes between the field and the field_slice match for axes not in the location

        data = field_slice.data

        idxs_tuple = self.__idxs_tuple(location)

        existing_data = self.data[idxs_tuple]

        unknown_mask = np.isnan(data)
        known_mask = np.invert(unknown_mask)

        existing_unknowns = np.isnan(existing_data[known_mask])
        overriding = not np.all(existing_unknowns)

        if allow_override and overriding:
            print(f"Boundary conditions and initial conditions may be in conflict for field {field_slice.name}\n\
Conflicting values override and become equal to whatever was assigned last\n")
        elif not allow_override and overriding:
            raise Exception("attempting to override values, this may be due to too many boundary conditions.")

        data[unknown_mask] = existing_data[unknown_mask]

        self.data[idxs_tuple] = data

    def __get_data(self, location={}):

        # returns an n-dimensional numpy array of the data at the given location

        # check valid idx is overkill here since I always check it prior to passing it through but OK

        data = self.data
        data_slice = data[self.__idxs_tuple(location)]

        return FieldInstant(self.model, self.edge_axes, self.name, data_slice)

    def __idxs_tuple(self, idxs):

        # converts a dictionary of idxs to a tuple of indices which can be used to index the numpy array
        # {a:1,b:2} --> (1,2,:)            if axes are a,b,c

        # first checks if idx is in the form of a dictionary like {"a":1,"b":2} where they're all axis less than their lengths

        assert type(idxs) == dict, "indexing must be a dictionary"

        axes_lengths = dict(
            zip(self.model.axes_names, self.model.axes_lengths))

        for axis in idxs:
            assert type(axis) == str, f"{axis} is not a string"
            assert axis in self.model.axes_names, f"{axis} is not an axis in the model"
            assert axes_lengths[axis] > idxs[axis], f"{axis} goes out of bounds, it has length {axes_lengths[axis]}"

        # then does the conversion

        idxs_filled = dict()
        for axis in self.model.axes:
            if axis not in idxs.keys():
                idxs_filled[axis] = slice(None)
            else:
                idxs_filled[axis] = idxs[axis]

        return tuple(idxs_filled[axis] for axis in self.model.axes)

    def imshow(self, location={}):

        # noninteractive visual of field

        # a bit redundant with the interact method (but implemented differently)
        # why keep this:
        # can run without notebook
        # more general (location can be for any set of axes, not just time)

        location_axes = list(location.keys())

        im_axes = [
            axis for axis in self.model.axes_names if axis not in location_axes]

        if len(im_axes) == 0:
            raise Exception("no remaining data once location is specified")
        elif len(im_axes) == 1:
            raise Exception(
                "only 2-dimensional data allowed, use plot for 1-dimensional")
        elif len(im_axes) > 2:
            raise Exception(
                "cannot suppport more than 2-dimensional data for imshow")

        x_axis = im_axes[0]
        y_axis = im_axes[1]

        axes_values = dict(zip(self.model.axes_names, self.model.axes_values))

        def get_bounds(axis):
            axis_values = axes_values[axis]
            return [min(axis_values), max(axis_values)]

        bounds = map(get_bounds, im_axes)

        bounds_flat = [val for bound in bounds for val in bound]

        # why transpose? ...
        # for the original data
        #   the higher priority axis (which comes first) is the rows --> y axis for imshow
        #   the lower priority axis (which comes second) is the columns --> x axis for imshow
        # BUT bounds go x values first then y values
        # therefore I need to take the transpose so the two match

        # why flipped up down?...
        # when tranposing, columns to the right become rows lower down
        # but when plotting rows lower down mean lower values --> need to flip them to the top

        im_data0 = self.__get_data(location).data

        im_data = np.flipud(np.transpose(im_data0))

        plt.imshow(im_data, extent=bounds_flat)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

    def plot(self, location={}):

        # as with imshow, noninteractive visual of field

        # a bit redundant with the interact method (but implemented differently)
        # why keep this:
        # can run without notebook
        # more general (location can be for any set of axes, not just time)

        plot_data = self.__get_data(location).data

        location_axes = list(location.keys())

        plot_axes = [
            axis for axis in self.model.axes_names if axis not in location_axes]

        assert len(plot_axes) == 1, "only 1-dimensional data allowed for plot"

        axes_values = dict(zip(self.model.axes_names, self.model.axes_values))

        x_axis = plot_axes[0]

        x_values = axes_values[x_axis]

        plt.plot(x_values, plot_data)
        plt.xlabel(x_axis)

    def __str__(self) -> str:
        return f"{len(self.model.axes_names)}-dimensional Field named {self.name}, dimension lengths: {dict(zip(self.model.axes_names,self.model.axes_lengths))}"

    @staticmethod
    def __field_op():
        raise TypeError(
            "Cannot perform arithmetic between fields directly. First use the \
prev or new properties to get the fields at the current timestep.")

    def __neg__(self):
        Field.__field_op()

    def __add__(self, other):
        Field.__field_op()

    def __sub__(self):
        Field.__field_op()

    def __mul__(self):
        Field.__field_op()

    def __truediv__(self):
        Field.__field_op()

    def __pow__(self):
        Field.__field_op()

    def __radd__(self):
        Field.__field_op()

    def __rsub__(self):
        Field.__field_op()

    def __rmul__(self):
        Field.__field_op()

    def __rtruediv__(self):
        Field.__field_op()

    def __rpow__(self):
        Field.__field_op()

class ConstantField(Field):

    def __init__(self, model: Model, name: str, edge_axes: tuple = ()):

        """
        A Field which does not change with time

        model: a Model object which the field is part of, all fields should share the same model
        name: field name, used for labeling plots
        edge_axes: the axes along which the values of the field are at the edges, between cells
        """
        super().__init__(model, name, edge_axes, 0)
        self.updated = True

    @property
    def prev(self):
        raise Exception("prev property not allowed for ConstantField, use always property instead")

    @property
    def always(self):
        time_axis = self.model.time_axis
        return self._Field__get_data({time_axis: 0}) # could choose any time

    def set_IC(self,_):
        raise Exception("set_IC not allowed for ConstantField, use set instead")
    
    def set_BC(self,_):
        raise Exception("set_BC not allowed for ConstantField, use set instead")
    
    def set(self,expression):
        # TODO this still allows them to make the field time dependent, it's ok though
        self._Field__set_expression(expression)

class FieldInstant:

    def __init__(self, model, edge_axes, name, data):

        """
        field at one instant of time
        constructed by new and prev method, \
            and required for update and update_time_integrate methods
        not recommended to be constructed by user directly
        """

        self.model = model
        self.edge_axes = edge_axes
        self.name = name
        self.data = data

    def __str__(self) -> str:
        return f"{len(self.model.axes_names)}-dimensional field at the current time,\n\
Dimension lengths: {dict(zip(self.model.axes_names,self.model.axes_lengths))}"

    def __field_op(op1, op2, operation):

        def get_operand_data(operand):
            # both extracts operand data and adds the models for later comparison
            if isinstance(operand, Field):
                raise TypeError(
                    "Cannot perform arithmetic between a field and a field at a given moment -- use the new or prev properties for both fields")
            if isinstance(operand, FieldInstant):
                models.append(operand.model)
                return operand.data
            elif isinstance(operand, numbers.Number):
                return operand
            else:
                raise TypeError(
                    "can only perform arithmetic between fields or between fields and numbers")

        models = []

        if op2 == None:
            # case for single argument, i think just negation
            new_data = operation(op1.data)

        else:
            operand_data = list(map(get_operand_data, [op1, op2]))
            # new_field.data = operation(get_operand_data(op1),get_operand_data(op2))

            both_fields = len(models) == 2

            if both_fields and models[0] != models[1]:
                raise ValueError(
                    "can only perform arithmetic operations between fields with the same model")

            elif both_fields and set(op1.edge_axes) != set(op2.edge_axes):
                raise Exception(
                    "Fields must share the same axes that are on edges vs on cells")

            new_data = operation(operand_data[0], operand_data[1])

        new_field = FieldInstant(op1.model, op1.edge_axes, op1.name, new_data)

        return new_field

    def __neg__(self):
        return FieldInstant.__field_op(self, None, operator.neg)

    def __add__(self, other):
        return FieldInstant.__field_op(self, other, operator.add)

    def __sub__(self, other):
        return FieldInstant.__field_op(self, other, operator.sub)

    def __mul__(self, other):
        return FieldInstant.__field_op(self, other, operator.mul)

    def __truediv__(self, other):
        assert not isinstance(other, Field), "cannot divide by field"
        return FieldInstant.__field_op(self, other, operator.truediv)

    def __pow__(self, other):
        assert not isinstance(other, Field), "cannot raise to field"
        return FieldInstant.__field_op(self, other, operator.pow)

    def __radd__(self, other):
        return FieldInstant.__field_op(self, other, operator.add)

    def __rsub__(self, other):
        return FieldInstant.__field_op(self, other, operator.sub)

    def __rmul__(self, other):
        return FieldInstant.__field_op(self, other, operator.mul)

    def __rtruediv__(self, other):
        raise Exception("cannot divide by field")

    def __rpow__(self, other):
        raise Exception("cannot raise to field")

class Stencil:

    def __init__(self, sample_points: list, der_order: int, axis_type:str="cell", der_axis_type:str="cell"):

        """
        creates a finite difference stencil at the given coordinates and for the given derivative order
        automatically computes the weights for each coordinate
        can take derivatives of the Fields using the stencil
        use to_text to see the approximation
        
        sample_points: the coordinates for creating the stencil
        der_order: the order of the derivative
        axis_type: either "cell" or "edge", whether the coordinates used are on the cells or the edges between cells
        der_axis_type: either "cell" or "edge", whether the resulting derivative is on the cells or the edges between cells

        If axis_type and der_axis_type are the same, integer values for sample points must be used
        If they're different, half values must be used
        """

        assert type(
            sample_points) == list, "points must be a list of locations to sample from for computing the derivative"
        assert type(
            der_order) == int and der_order > 0, "derivative order must be a positive integer"
        assert len(set(sample_points)) == len(
            sample_points), "the coordinates of the points to sample must all be unique values"

        allowable_axis_types = ["cell", "edge"]

        assert axis_type in allowable_axis_types, f"axis_type must be one of {allowable_axis_types}"
        assert der_axis_type in allowable_axis_types, f"der_axis_type must be one of {allowable_axis_types}"

        if len(sample_points) <= der_order:
            raise Exception(
                "there must be at least one more point than the derivative order")

        from_edge = axis_type == "edge"
        to_edge = der_axis_type == "edge"

        points = np.array(sample_points)

        whole_values = np.all(points % 1 == 0)
        frac_values = np.all((points+0.5) % 1 == 0)

        if whole_values:
            intermediate = False
        elif frac_values:
            intermediate = True
        else:
            raise Exception(
                "points need to be all integers or all half values")

        if (from_edge and not to_edge) and not intermediate:
            raise Exception(
                "going from edges to cells requires stencil with half values in difference approximation")
        elif (from_edge and not to_edge) and not intermediate:
            raise Exception(
                "going from cells to edges requires stencil with half values in difference approximation")
        elif (from_edge and to_edge) and intermediate:
            raise Exception(
                "going from edges to edges requires stencil with integer values in difference approximation")
        elif (not from_edge and not to_edge) and intermediate:
            raise Exception(
                "going from cells to cells requires stencil with integer values in difference approximation")

        # constructs and solves taylor expansion matrix
        # uses delta half of the step size to allow for half values

        half_points = points*2

        M_size = len(half_points)

        coeff = np.matrix(half_points)
        powers = np.matrix(np.arange(0, M_size)).transpose()

        M = np.power(coeff, powers)

        b = np.zeros((M_size, 1))
        # 2^der comes from adjusting for the half step size
        b[der_order] = math.factorial(der_order)*2**der_order

        weights = np.around(np.linalg.inv(M)*b, 2).transpose().tolist()[0]

        self.intermediate = intermediate

        self.points = points
        self.weights = weights
        self.der_order = der_order

        self.from_edge = from_edge
        self.to_edge = to_edge

        self.to_text()

    def to_text(self):

        """
        prints the equation for the finite difference approximation 
        """

        expression_parts = []

        for coefficient, interval in zip(self.weights, self.points):
            expression_parts.append(f"{coefficient}f(x+{interval}h)")
        numerator = "+".join(expression_parts)
        denominator = f"h^{self.der_order}"
        der_marks = "".join(["'"]*self.der_order)
        derivative = f"f{der_marks}"

        equation = f"{derivative} = [{numerator}] / [{denominator}]"

        equation = equation.replace("+-", "-")
        equation = equation.replace(".0", "")
        equation = equation.replace("1f", "f")
        equation = equation.replace("1h", "h")
        equation = equation.replace(")+", ") + ")
        equation = equation.replace(")-", ") - ")

        print(f"Finite approximation: {equation}\n")

    def der(stencil, f:FieldInstant, der_axis: str):

        """
        returns the derivative of f at the current time using the stencil
        performs derivative operation along der_axis

        stencil: a Stencil object defining the numerical derivative
        f: the Field object to take the derivative of
        der_axis: the axis to take the derivative along
        """


        assert isinstance(f, FieldInstant), "Can only perform derivative on a field at a moment, use prev or new properties of the field"
        assert isinstance(stencil, Stencil), "stencil must be a Stencil object"
        assert type(der_axis) == str, "der_axis must be a string of the axis name"
        assert f.model.time_axis is not None, "cannot take derivative of a field with no time axis"
        
        non_time_axes = f.model.axes_names
        non_time_axes.remove(f.model.time_axis)

        assert der_axis in non_time_axes, f"der_axis must be a spatial axis, '{der_axis}' is not"

        axis_n = non_time_axes.index(der_axis)

        n_axes = len(non_time_axes)

        from_edge = der_axis in f.edge_axes

        if from_edge and not stencil.from_edge:
            raise Exception(
                f"Stencil takes a field with cells, but the field has edges along the {der_axis} axis")

        elif not from_edge and stencil.from_edge:
            raise Exception(
                f"Stencil takes a field with edges, but the field has cells along the {der_axis} axis")

        to_edge = stencil.to_edge

        shifts = stencil.points
        weights = stencil.weights

        if from_edge and not to_edge:
            # E -> C
            shifts = shifts + 1/2
            shape_shift = -1
        elif not from_edge and to_edge:
            # C -> E
            shifts = shifts - 1/2
            shape_shift = 1
        else:
            shape_shift = 0

        # the derivative is the same shape as the original function, except if there's a switch between edge and cell

        shifts = shifts.astype(int)

        periodic = der_axis in f.model.periodic

        if periodic:
            # for edge and periodic, it's just gonna be one less length, so can just roll now :) no removing and/or copying required

            # for the periodic case, number edges=number of cells
            derivative = np.zeros(f.data.shape)

            for i in range(len(shifts)):
                derivative += weights[i] * np.roll(f.data, shifts[i], axis=axis_n)

        # not periodic case:
        else:

            # shape only changes if it's not periodic
            fprime_shape = list(f.data.shape)
            fprime_shape[axis_n] += shape_shift
            fprime_shape = tuple(fprime_shape)

            derivative = np.full(fprime_shape, np.nan)

            if from_edge and not to_edge:
                # E->C means that one more cell on the right that could potentially be filled has to be eliminated if all the points are on the left side (shits[-1]<=0)
                allowable_max_shift = 1
            else:
                allowable_max_shift = 0

            min_shift = min(shifts[0], 0)
            max_shift = max(shifts[-1], allowable_max_shift)

            original_length = f.data.shape[axis_n]
            partial_length = original_length+min_shift-max_shift

            section_shape = list(fprime_shape)
            section_shape[axis_n] = partial_length
            section_derivative = np.zeros(section_shape)

            # need to pad current_data with nan on the right side if C -> E

            for i in range(len(shifts)):

                shift = shifts[i]
                lower_idx = shift-min_shift
                upper_idx = shift-min_shift+partial_length

                idxs = [slice(None)]*n_axes
                idxs[axis_n] = slice(lower_idx, upper_idx)
                idxs = tuple(idxs)

                section_derivative += weights[i]*f.data[idxs]

            lower_placement_idx = -min_shift
            upper_placement_idx = -min_shift+partial_length

            placement_idxs = [slice(None)]*n_axes
            placement_idxs[axis_n] = slice(
                lower_placement_idx, upper_placement_idx)
            placement_idxs = tuple(placement_idxs)

            derivative[placement_idxs] = section_derivative

        step_sizes = dict(zip(f.model.axes_names,f.model.axes_step_size))
        dx = step_sizes[der_axis]

        derivative /= dx**stencil.der_order

        if np.any(np.isinf(derivative)):
            raise Exception("Instability has led to infinite values, considering decreasing timestep")

        derivative_edge_axes = list(f.edge_axes)

        if stencil.from_edge and not stencil.to_edge:
            derivative_edge_axes.remove(der_axis)
        elif not stencil.from_edge and stencil.to_edge:
            derivative_edge_axes += der_axis

        return FieldInstant(f.model, derivative_edge_axes, f.name, derivative)


