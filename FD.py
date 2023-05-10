import numpy as np
import matplotlib.pyplot as plt
import operator
import numexpr
import itertools
import warnings
import numbers
import line_profiler
import math

# example of use: for wave equation deta/dx would be on the edges, then d/dx deta/dx would be back on the squares, bc can be applied to deta/dx on either side before differentiating again



class Model:
    # nonconstant property with current time? all operations then are only performed at that time (since the Field has access to Model props)
    def __init__(self,axes,periodic=[],time_axis=None):
        # dimensions is a dictionary with the dimension object and range

        def check_evenly_spaced_array(vals):
            try:
                iter(vals)
            except:
                raise TypeError("axes must be iterable")
            
            if not np.all([isinstance(val,numbers.Number) for val in vals]):
                raise TypeError("all values of axes must be numbers")

            diffs = np.diff(vals)
            tolerance = 10**-9 # numpy linspace isn't perfectly spaced
            if not np.all([abs(diffs[0]-diff)<tolerance for diff in diffs]):
                raise ValueError("all values of axes must be evenly spaced")
            
        assert set(periodic).issubset(set(axes)), f"{periodic} not a dimension"

        if time_axis!=None:
            assert type(time_axis)==str, "time axis must be a string"
            assert time_axis in axes, "time axis must be one of the axes"
            assert time_axis not in periodic, "time axis cant be periodic"

        assert type(axes)==dict, "axes must be input dictionary of dimensions and range of values"

        for i in range(len(axes)):
            axis_name = list(axes.keys())[i]
            axis_values = list(axes.values())[i]
            assert type(axis_name)==str, "axes keys must be strings, the name of the axis"
            check_evenly_spaced_array(axis_values)
            axes[axis_name] = np.array(axis_values) 


        self.axes = axes
        self.periodic = periodic
        self.time_step = 0 # this value steps each time
        self.time_axis = time_axis
        self.fields = []
        

        # would also have information like periodicity

    def increment_time(self):
        # TODO give fields names so they can be referenced here in the error message


        def clear_update(field):
            field.updated = False
            if field.dot!=None:
                clear_update(field.dot)

        for field in self.fields:
            assert field.updated, "not all fields have been updated"
            clear_update(field)

        self.time_step+=1

    @property
    def finished(self):
        time_values = self.axes[self.time_axis]
        n_steps = len(time_values)
        if self.time_step>=n_steps:
            raise ValueError("the current timestep reached the number of timestips -- this shouldn't happen")
        return self.time_step+1==n_steps

    @property
    def dt(self):
        assert self.time_axis!=None, "no time axis"
        return dict(zip(self.axes_names,self.axes_step_size))[self.time_axis]


    @property
    def n_time_steps(self):
        assert self.time_axis!=None, "no time axis"
        return dict(zip(self.axes_names,self.axes_lengths))[self.time_axis]
    
    
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



class Stencil:
    # could construct it as a dictionary with numbers for keys --> allows for negative "indexing"
    def __init__(self,sample_points,der_order,axis_type="cell",der_axis_type="cell"):#from_edge = False,to_edge =False):

        

        assert type(sample_points)==list, "points must be a list of locations to sample from for computing the derivative"
        assert type(der_order)==int and der_order>0, "derivative order must be a positive integer"


        allowable_axis_types = ["cell","edge"]

        assert axis_type in allowable_axis_types, f"axis_type must be one of {allowable_axis_types}"
        assert der_axis_type in allowable_axis_types, f"der_axis_type must be one of {allowable_axis_types}"


        if len(sample_points)<=der_order:
            raise Exception("there must be at least one more points than the derivative order")



        from_edge = axis_type=="edge"
        to_edge = der_axis_type=="edge"


        points = np.array(sample_points)

        whole_values = np.all(points%1==0)
        frac_values = np.all((points+0.5)%1==0)

        if whole_values:
            intermediate = False
        elif frac_values:
            intermediate = True
        else:
            raise Exception("points need to be all integers or all half values")



        if (from_edge and not to_edge) and not intermediate:
            raise Exception("going from edges to cells requires stencil with half values in difference approximation")
        elif (from_edge and not to_edge) and not intermediate:
            raise Exception("going from cells to edges requires stencil with half values in difference approximation")
        elif (from_edge and to_edge) and intermediate:
            raise Exception("going from edges to edges requires stencil with integer values in difference approximation")
        elif (not from_edge and not to_edge) and intermediate:
            raise Exception("going from cells to cells requires stencil with integer values in difference approximation")



        half_points = points*2

        M_size = len(half_points)

        coeff = np.matrix(half_points)
        powers = np.matrix(np.arange(0,M_size)).transpose()

        M = np.power(coeff,powers)

        b = np.zeros((M_size,1))
        b[der_order] = math.factorial(der_order)*2**der_order # 2^der comes from adjusting for the half step size

        weights = np.around(np.linalg.inv(M)*b,2).transpose().tolist()[0]


        self.intermediate = intermediate
        
        self.points = points
        self.weights = weights
        self.der_order = der_order

        self.from_edge = from_edge
        self.to_edge = to_edge
        
        self.to_text()



    def to_text(self):

        expression_parts = []

        # Iterate over the coefficients and intervals simultaneously
        for coefficient, interval in zip(self.weights, self.points):
            expression_parts.append(f"{coefficient}f(x+{interval}h)")

        # Join the expression parts with a plus sign
        numerator = "+".join(expression_parts)

        denominator = f"h^{self.der_order}"

        der_marks = "".join(["'"]*self.der_order)
        derivative = f"f{der_marks}"

        # Wrap the expression in LaTeX delimiters

        equation = f"{derivative} = [{numerator}] / [{denominator}]"

        equation = equation.replace("+-","-")
        equation = equation.replace(".0","")
        equation = equation.replace("1f","f")
        equation = equation.replace("1h","h")
        equation = equation.replace(")+",") + ")
        equation = equation.replace(")-",") - ")

        print(f"Finite approximation: {equation}")


    def der(stencil,f,der_axis):
        # returns the derivative of f at the current time using the stencil
        # performs derivative operation along der_axi

        
        assert isinstance(f,FieldInstant), "Can only perform derivative on a field at a moment, use prev or new properties of the field"
        assert isinstance(stencil,Stencil), "stencil must be a Stencil object"
        assert f.model.time_axis!=None, "why would you update derivative with no time axis"

        
        non_time_axes = f.model.axes_names
        non_time_axes.remove(f.model.time_axis)

        axis_n = non_time_axes.index(der_axis)

        n_axes = len(non_time_axes)


        from_edge = der_axis in f.edge_axes

        # TODO error message is a bit confusing since a field can be edge along one axis but cell along another
        if from_edge and not stencil.from_edge:
            raise Exception("Stencil takes a cell field but an edge field was given")
        
        elif not from_edge and stencil.from_edge:
            raise Exception("Stencil takes an edge field but a cell field was given")
        

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



            derivative = np.zeros(f.data.shape) # for the periodic case, number edges=number of cells

            for i in range(len(shifts)):
                derivative+=weights[i]*np.roll(f.data,shifts[i],axis=axis_n)


        # not periodic case:
        else:

            # shape only changes if it's not periodic
            fprime_shape = list(f.data.shape)
            fprime_shape[axis_n]+=shape_shift
            fprime_shape = tuple(fprime_shape)

            
            derivative = np.full(fprime_shape,np.nan)


            if from_edge and not to_edge:
                # E->C means that one more cell on the right that could potentially be filled has to be eliminated if all the points are on the left side (shits[-1]<=0)
                allowable_max_shift  = 1
            else:
                allowable_max_shift = 0

            min_shift = min(shifts[0],0)
            max_shift = max(shifts[-1],allowable_max_shift)

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
                idxs[axis_n] = slice(lower_idx,upper_idx)
                idxs = tuple(idxs)

                section_derivative+=weights[i]*f.data[idxs]

            lower_placement_idx = -min_shift
            upper_placement_idx = -min_shift+partial_length


            placement_idxs = [slice(None)]*n_axes
            placement_idxs[axis_n] = slice(lower_placement_idx,upper_placement_idx)
            placement_idxs = tuple(placement_idxs)

            derivative[placement_idxs] = section_derivative


        derivative_edge_axes = f.edge_axes.copy()


        if stencil.from_edge and not stencil.to_edge:
            derivative_edge_axes.remove(der_axis)
        elif not stencil.from_edge and stencil.to_edge:
            derivative_edge_axes += der_axis

        return FieldInstant(f.model,derivative_edge_axes,derivative)



class Field:

    def __init__(self,model,edge_axes = [],n_time_ders=0):


        assert isinstance(model,Model), "model must be a Model object"

        axes = model.axes_names

        if type(edge_axes)==str:
            edge_axes = [edge_axes]

        assert set(edge_axes)-set(axes)==set(), "edge_axes must be axes in the model"

        assert type(n_time_ders)==int and n_time_ders>=0, "n_time_ders must be an integer greater than equal to 0"

        if n_time_ders>0:
            self.dot = Field(model,edge_axes,n_time_ders-1)
        else:
            self.dot = None

        model.fields.append(self) # model has a list of all the fields


        axes_lengths = dict(zip(model.axes_names,model.axes_lengths))

        for axis in edge_axes:
            if axis not in model.periodic:
                axes_lengths[axis]+=1

        axes_lengths = tuple(axes_lengths.values())

        self.model = model
        self.edge_axes = edge_axes
        self.data = np.full(axes_lengths,np.nan)
        self.updated = False

    @property
    def new(self):
        # returns an n-dimensional numpy array of all the data at the current time
        assert self.updated, "field has not yet been updated, use prev property to get current values"
        current_time = self.model.time_step+1
        time_axis = self.model.time_axis
        return self.__get_data({time_axis:current_time})
    
    @property
    def prev(self):
        # returns an n-dimensional numpy array of all the data at the following time
        # this is used for time integration
        assert not self.updated, "field has already been updated, use new property to get values at the next timestep"
        previous_time = self.model.time_step
        time_axis = self.model.time_axis
        return self.__get_data({time_axis:previous_time})
    


    def set_IC(self,expression):
        assert self.dot!=None, "field must have a time derivative to set initial conditions"
        time_axis = self.model.time_axis
        self.__set_expression(expression,{time_axis:0})

    def set_BC(self,expression,axis,side):

        assert axis!=self.model.time_axis, "cannot use BC to set time axis, use set_IC instead"
        assert axis in self.model.axes_names, "axis must be one of the axes names"

        side_types = ["start","end"]
        assert side in side_types, f"side must be one of {side_types}"

        if side=="start":
            idx = 0
        elif side=="end":
            idx = -1

        self.__set_expression(expression,{axis:idx})

    def __set_expression(self,expression,location={}):
        # sets the data to the value of the expression at the specified location
        # used for boundary conditions and initial conditions

        assert type(expression)==str, "expression must be a string"





        def transpose(nested_list):
            transposed_list = [[row[i] for row in nested_list] for i in range(len(nested_list[0]))]
            return transposed_list

        substitute_axes_names = []
        substitute_axes_values = []
        substitute_axes_lengths = []

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
                step_sizes = dict(zip(self.model.axes_names,self.model.axes_step_size))
                dx = step_sizes[axis_name]


                axis_values = [val-dx/2 for val in axis_values]



                
            if edge_axis and not periodic:
                # for edge axis that aren't periodic, there's an extra value at the end (5 cells means 6 edges)
                # if it's periodic, it's the same amount since the two edges on the end are the same
                final_value = axis_values[-1]
                axis_values.append(final_value+dx)
                axis_length+=1


            substitute_axes_names.append(axis_name)
            substitute_axes_values.append(axis_values)
            substitute_axes_lengths.append(axis_length)





        # constructs nested list of all combinations of values
        value_combs = transpose([list(comb) for comb in itertools.product(*substitute_axes_values)])
        subs = dict(zip(substitute_axes_names,value_combs))

        # TODO: use more granular check, first checking if all the variables in the expression are axes, may require sympy
        try:
            data_flat = numexpr.evaluate(expression,subs)
        except:
            raise Exception("either variable mismatch or cannot parse expression")


        # if the expression is just a constant, numexpr just returns a single value instead of an array
        if data_flat.shape==():
            n_subs = len(value_combs[0])
            data_flat = np.full((n_subs,1),float(data_flat))
        
        data = np.reshape(data_flat,substitute_axes_lengths)

        field_slice = FieldInstant(self.model,self.edge_axes,data)

        self.__set_data(field_slice,location,allow_override=True) # allow override allows you to (for example) override boundary conditions with initial conditions or vice versa



    def assign_update(self,values):

        assert not self.updated, "field already updated"
        assert self.dot==None, "cannot use assign_update on field with time derivative, use time_integrate_update instead"
        assert isinstance(values,FieldInstant)

        time_axis = self.model.time_axis
        time = self.model.time_step+1

        self.__set_data(values,{time_axis:time},allow_override=False)
        self.updated = True


    def time_integrate_update(self):
        # updates values using time integration

        assert not self.updated, "field already updated"
        assert self.dot!=None, "field needs to have time derivative to perform time_integrate"
        assert self.dot.updated, "time derivative needs to be updated first"


        dt = self.model.dt
        new_slice = self.prev + self.dot.new*dt

        # TODO more than just euler's method, allow runge kutta as well

        time_axis = self.model.time_axis
        time = self.model.time_step+1 # +1 since it's updating the following value

        self.__set_data(new_slice,{time_axis:time},allow_override=False)
        self.updated = True


        if np.any(np.isnan(self.new.data)):
            raise Exception("unknown values of field after time integration")






    def __set_data(self,field_slice,location,allow_override):

        # combines self's data at the location with the input data (merging unknown and known values)
        # then sets self's data to merged data


        assert isinstance(field_slice,FieldInstant),"must input field slice"
        data = field_slice.data

        idxs_tuple =  self.__idxs_tuple(location)


        existing_data = self.data[idxs_tuple] 



        unknown_mask = np.isnan(data)
        known_mask = np.invert(unknown_mask)


        existing_unknowns = np.isnan(existing_data[known_mask])
        overriding = not np.all(existing_unknowns)

        if allow_override and overriding:
            # override allowed when bc overrides initial conditions or vice versa (initial setup)
            warnings.warn("Overriding values",category=Warning)
        elif not allow_override and overriding:
            raise Exception("attempting to override values")



        data[unknown_mask] = existing_data[unknown_mask]



        self.data[idxs_tuple]  = data





    def __get_data(self,location={}):
        

        # TODO make it return an object of a class NowField

        # returns an n-dimensional numpy array of the data at the given location 

        # check valid idx is overkill here since I always check it prior to passing it through but OK

        data = self.data
        data_slice = data[self.__idxs_tuple(location)]

        return FieldInstant(self.model,self.edge_axes,data_slice)





    
    def __idxs_tuple(self,idxs):


        # converts a dictionary of idxs to a tuple of indices which can be used to index the numpy array
        # {a:1,b:2} --> (1,2,:)            if axes are a,b,c



        # first checks if idx is in the form of a dictionary like {"a":1,"b":2} where they're all axis less than their lengths
        
        assert type(idxs)==dict, "indexing must be a dictionary"

        axes_lengths = dict(zip(self.model.axes_names,self.model.axes_lengths))

        for axis in idxs:
            assert type(axis)==str, f"{axis} is not a string"
            assert axis in self.model.axes_names, f"{axis} is not an axis in the model"
            assert axes_lengths[axis]>idxs[axis], f"{axis} goes out of bounds, it has length {axes_lengths[axis]}"


        # then does the conversion


        idxs_filled = dict()
        for axis in self.model.axes:
            if axis not in idxs.keys():
                idxs_filled[axis] = slice(None)
            else:
                idxs_filled[axis] = idxs[axis]

        return tuple(idxs_filled[axis] for axis in self.model.axes)

    def imshow(self,location={}):

        

        im_data = self.__get_data(location).data



        location_axes = list(location.keys())

        im_axes = [axis for axis in self.model.axes_names if axis not in location_axes]

        if len(im_axes)==0:
            raise Exception("no remaining data once location is specified")
        elif len(im_axes)==1:
            raise Exception("only 2-dimensional data allowed, use plot for 1-dimensional")
        elif len(im_axes)>2:
            raise Exception("cannot suppport more than 2-dimensional data for imshow")
        
        x_axis = im_axes[0]
        y_axis = im_axes[1]

        axes_values = dict(zip(self.model.axes_names,self.model.axes_values))

        def get_bounds(axis):
            axis_values = axes_values[axis]
            return [min(axis_values),max(axis_values)]
        
        bounds = map(get_bounds,im_axes)

        bounds_flat = [val for bound in bounds for val in bound]

        plt.imshow(im_data,extent=bounds_flat)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
    
    def plot(self,location={}):


        plot_data = self.__get_data(location).data


        location_axes = list(location.keys())

        plot_axes = [axis for axis in self.model.axes_names if axis not in location_axes]

        assert len(plot_axes)==1, "only 1-dimensional data allowed for plot"


        axes_values = dict(zip(self.model.axes_names,self.model.axes_values))

        x_axis = plot_axes[0]


        x_values = axes_values[x_axis]




        plt.plot(x_values,plot_data)
        plt.xlabel(x_axis)

    def __str__(self) -> str:
        return f"{len(self.model.axes_names)}-dimensional Field, dimension lengths: {dict(zip(self.model.axes_names,self.model.axes_lengths))}"
       
    

    def __field_op(op1,op2,op):

        raise TypeError("Cannot perform arithmetic between fields directly. First use the prev or new properties to get the fields at the current timestep.")


    # TODO doesn't need to take the operation as an argument, nor does it need to return anything

    def __neg__(self):
        return Field.__field_op(self,None,operator.neg)
    
    def __add__(self,other):
        return self.__field_op(other,operator.add)
    
    def __sub__(self,other):
        return Field.__field_op(self,other,operator.sub)
    
    def __mul__(self,other):
        return Field.__field_op(self,other,operator.mul)

    def __truediv__(self,other):
        assert not isinstance(other,Field), "cannot divide by field"
        return Field.__field_op(self,other,operator.truediv)
   
    def __pow__(self,other):
        assert not isinstance(other,Field), "cannot raise to field"
        return Field.__field_op(self,other,operator.pow)
    
    def __radd__(self,other):
        return Field.__field_op(self,other,operator.add)
    
    def __rsub__(self,other):
        return Field.__field_op(self,other,operator.sub)
    
    def __rmul__(self,other):
        return Field.__field_op(self,other,operator.mul)
    
    def __rtruediv__(self,other):
        return Field.__field_op(self,other,operator.mul)
    
    def __rpow__(self,other):
        return Field.__field_op(self,other,operator.mul)
    


class FieldInstant:
    # field at one instant of time
    # constructed by now and next method
    # required for update and update_time_integrate methods
    def __init__(self,model,edge_axes,data):
        # I don't think n_time_ders is needed for this
        self.model = model
        self.edge_axes = edge_axes
        self.data = data

    def __field_op(op1,op2,operation):

        def get_operand_data(operand):
            # both extracts operand data and adds the models for later comparison
            if isinstance(operand,Field):
                raise TypeError("Cannot perform arithmetic between a field and a field at a given moment -- use the new or prev properties for both fields")
            if isinstance(operand,FieldInstant):
                models.append(operand.model)
                return operand.data
            elif isinstance(operand,numbers.Number):
                return operand
            else:
                raise TypeError("can only perform arithmetic between fields or between fields and numbers")


        models = []




        if op2==None:
            # case for single argument, i think just negation
            new_data = operation(op1.data)
        

        else:
            operand_data = list(map(get_operand_data,[op1,op2]))
            #new_field.data = operation(get_operand_data(op1),get_operand_data(op2))


            both_fields = len(models)==2

            if both_fields and models[0]!=models[1]:
                raise ValueError("can only perform arithmetic operations between fields with the same model")

            elif both_fields and set(op1.edge_axes)!=set(op2.edge_axes):
                raise Exception("Fields must share the same axes that are on edges vs on cells")


            new_data = operation(operand_data[0],operand_data[1])



        new_field = FieldInstant(op1.model,op1.edge_axes,new_data)

    
        return new_field
    


       
    def __neg__(self):
        return FieldInstant.__field_op(self,None,operator.neg)
    
    def __add__(self,other):
        return FieldInstant.__field_op(self,other,operator.add)
    
    def __sub__(self,other):
        return FieldInstant.__field_op(self,other,operator.sub)
    
    def __mul__(self,other):
        return FieldInstant.__field_op(self,other,operator.mul)

    def __truediv__(self,other):
        assert not isinstance(other,Field), "cannot divide by field"
        return FieldInstant.__field_op(self,other,operator.truediv)
   
    def __pow__(self,other):
        assert not isinstance(other,Field), "cannot raise to field"
        return FieldInstant.__field_op(self,other,operator.pow)
    
    def __radd__(self,other):
        return FieldInstant.__field_op(self,other,operator.add)
    
    def __rsub__(self,other):
        return FieldInstant.__field_op(self,other,operator.sub)
    
    def __rmul__(self,other):
        return FieldInstant.__field_op(self,other,operator.mul)
    
    def __rtruediv__(self,other):
        raise Exception("cannot divide by field")
    
    def __rpow__(self,other):
        raise Exception("cannot raise to field")