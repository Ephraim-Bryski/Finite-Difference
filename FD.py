import numpy as np
import matplotlib.pyplot as plt
import operator
import numexpr
import itertools
import warnings
import numbers
import line_profiler

# example of use: for wave equation deta/dx would be on the edges, then d/dx deta/dx would be back on the squares, bc can be applied to deta/dx on either side before differentiating again



class Domain:
    # nonconstant property with current time? all operations then are only performed at that time (since the Field has access to Domain props)
    def __init__(self,axes,periodic=[],time_axis=None,check_bc=True):
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

        # if check_c is True, it doesn't allow you to override boundary conditions, throws error
        # if check_bc is False, doesn't throw error, but is much faster
        # probably want to initially test with check_bc on, then turn check_bc off for performance
        assert type(check_bc)==bool, "strict must be a boolean"

        for i in range(len(axes)):
            axis_name = list(axes.keys())[i]
            axis_values = list(axes.values())[i]
            assert type(axis_name)==str, "axes keys must be strings, the name of the axis"
            check_evenly_spaced_array(axis_values)
            axes[axis_name] = np.array(axis_values) 


        self.axes = axes
        self.periodic = periodic
        self.time = 0 # this value steps each time
        self.time_axis = time_axis
        self.check_bc = check_bc
        

        # would also have information like periodicity

    # TODO: update_time should also check if there are no None values for the derivatives at the given time
    def update_time(self,fs):
        # fs is a tuple of functions and their derivatives
        # in order (fppp,fpp,fp,f)
        # for now just assuming euler time step
        assert type(self.time_axis)==str, "need time axis to increment time"
        assert type(fs)==tuple, "must input a tuple of fields"
        assert len(fs)>1, "must be at least two fields in tuple"
        assert np.all([isinstance(f,Field) for f in fs]), " all elements must be fields"
        assert np.all([self == fs[0].domain for f in fs]), "all fields must share the domain"

        dt = self.dt

        current = {self.time_axis:self.time}
        next = {self.time_axis:self.time+1}

        for i in range(len(fs)-1):
            
            if i==0:
                der_loc = current
            else:
                der_loc = next

            fnew = fs[i+1].get_data(current)+fs[i].get_data(der_loc)*dt

            fs[i+1].set_data(fnew,next,allow_override=False)
        
        self.time+=1



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




class Kernel:
    # could construct it as a dictionary with numbers for keys --> allows for negative "indexing"
    def __init__(self,values,center_idx,der_order,axis,domain):
            # would then use the axes_step_size property to construct the kernel values
        assert type(values)==list, "values must be a list"
        assert type(center_idx)==int, "central index must be an integer"
        assert type(der_order)==int and der_order>0, "order of the derivative must be a positive integer"
        assert axis in domain.axes_names, "axis must be in the domain"
        assert len(values)>center_idx, "center_idx must be an index within the kernel size"
        
        kernel = dict()
        for i in range(len(values)):
            value = values[i]
            shifted_idx = i-center_idx
            kernel[shifted_idx] = value
        self.values = kernel
        self.axis = axis
        self.der_order = der_order # used for determining what power to raise dx to

  
class Unknown:
    def __init__(self):
        pass

    def __repr__(self):
        return "?"

    def __neg__(self):
        return self
        
    def __add__(self,_):
        return self
    
    def __sub__(self,_):
        return self
        
    def __mul__(self,_):
        return self

    def __truediv__(self,_):
        return self
   
    def __pow__(self,_):
        return self
    
    def __radd__(self,_):
        return self
    
    def __rsub__(self,_):
        return self
    
    def __rmul__(self,_):
        return self
    
    def __rtruediv__(self,_):
        raise self
    
    def __rpow__(self,_):
        raise self

    def __lt__(self,_):
        return False

class Field:

    def __init__(self,domain):
        assert isinstance(domain,Domain), "domain must be a Domain object"
        self.domain = domain


        # computations on object arrays are about 30 times slower, even with the same data
        # Unknown and nan behaves the same, nan just prevents you from checking if data is overriding
        if domain.check_bc:
            self.data = np.full(self.domain.axes_lengths,Unknown(),dtype="object")
        else:
            self.data = np.full(self.domain.axes_lengths,np.nan)

    def set_expression(self,expression,location={}):
        
        # sets the data to the value of the expression at the specified location
        # used for boundary conditions and initial conditions

        self.__check_valid_idx(location)
        assert type(expression)==str, "expression must be a string"


        def transpose(nested_list):
            transposed_list = [[row[i] for row in nested_list] for i in range(len(nested_list[0]))]
            return transposed_list

        substitute_axes_names = []
        substitute_axes_values = []
        substitute_axes_lengths = []

        for i in range(len(self.domain.axes_names)):
            axis_name = self.domain.axes_names[i]
            axis_values = self.domain.axes_values[i]
            axis_lengths = self.domain.axes_lengths[i]

            if axis_name not in location.keys():
                substitute_axes_names.append(axis_name)
                substitute_axes_values.append(axis_values)
                substitute_axes_lengths.append(axis_lengths)

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

        self.set_data(data,location,allow_override=True) # allow override allows you to (for example) override boundary conditions with initial conditions or vice versa

    def update_values(self,values):
        time_axis = self.domain.time_axis
        self.data[self.__idxs_tuple({time_axis:-1})] = values

    def update_der(self,f,kernel):
        # updates self by taking the derivative of f using the kernel
        # only performs the calculations at the current time

        assert isinstance(f,Field), "f must be a Field object"
        assert isinstance(kernel,Kernel), "kernel must be a Kernel object"
        assert self.domain==f.domain, "derivative and function must have the same domain"


        current_time = {self.domain.time_axis:self.domain.time}

        non_time_axes = self.domain.axes_names
        non_time_axes.remove(self.domain.time_axis)

        axis_n = non_time_axes.index(kernel.axis)

        n_axes = len(non_time_axes)


        axes_lengths = dict(zip(self.domain.axes_names,self.domain.axes_lengths))
        kernel_axis_len = axes_lengths[kernel.axis]

        shifts = list(kernel.values.keys())
        weights = list(kernel.values.values())


        current_data = f.data[f.__idxs_tuple(current_time)]


        if self.domain.check_bc:
            diff = np.zeros(current_data.shape,dtype="object")
        else:
            diff = np.zeros(current_data.shape)

        for i in range(len(kernel.values)):
            diff+=weights[i]*np.roll(current_data,shifts[i],axis=axis_n)

        step_sizes = dict(zip(self.domain.axes_names,self.domain.axes_step_size))
        dx = step_sizes[kernel.axis]

    

        diff/=dx**kernel.der_order


        left_cut = -shifts[0]
        right_cut = shifts[-1]


        start_idxs = [slice(None)]*n_axes
        start_idxs[axis_n] = slice(0,left_cut)

        end_idxs = [slice(None)]*n_axes
        end_idxs[axis_n] = slice(kernel_axis_len-right_cut,kernel_axis_len)

        periodic = kernel.axis in self.domain.periodic

        if not periodic:            
            if self.domain.check_bc:
                pad = Unknown()
            else:
                pad = np.nan

            diff[tuple(start_idxs)] = pad
            diff[tuple(end_idxs)] = pad

        self.set_data(diff,current_time,allow_override=False)




    @staticmethod
    def __convert_plot_data(data):
        # replace Unknowns with NaN so it can show it
        def replace_unknowns(val):
            if isinstance(val,Unknown):
                return np.nan
            else:
                return val
            
        data = np.vectorize(replace_unknowns)(data)
            
        return data.astype(float)

    def imshow(self,location={}):



        self.__check_valid_idx(location)

        im_data = self.get_data(location)

        if self.domain.check_bc:
            im_data = Field.__convert_plot_data(im_data)


        location_axes = list(location.keys())

        im_axes = [axis for axis in self.domain.axes_names if axis not in location_axes]

        if len(im_axes)==0:
            raise Exception("no remaining data once location is specified")
        elif len(im_axes)==1:
            raise Exception("only 2-dimensional data allowed, use plot for 1-dimensional")
        elif len(im_axes)>2:
            raise Exception("cannot suppport more than 2-dimensional data for imshow")
        
        x_axis = im_axes[0]
        y_axis = im_axes[1]

        axes_values = dict(zip(self.domain.axes_names,self.domain.axes_values))

        def get_bounds(axis):
            axis_values = axes_values[axis]
            return [min(axis_values),max(axis_values)]
        
        bounds = map(get_bounds,im_axes)

        bounds_flat = [val for bound in bounds for val in bound]

        plt.imshow(im_data,extent=bounds_flat)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
    
    def plot(self,location={}):

        self.__check_valid_idx(location)

        plot_data = self.get_data(location)

        if self.domain.check_bc:
            plot_data = Field.__convert_plot_data(plot_data)

        location_axes = list(location.keys())

        plot_axes = [axis for axis in self.domain.axes_names if axis not in location_axes]

        assert len(plot_axes)==1, "only 1-dimensional data allowed for plot"


        axes_values = dict(zip(self.domain.axes_names,self.domain.axes_values))

        x_axis = plot_axes[0]


        x_values = axes_values[x_axis]




        plt.plot(x_values,plot_data)
        plt.xlabel(x_axis)


    def get_data(self,location={}):
        
        # returns an n-dimensional numpy array of the data at the given location 

        # check valid idx is overkill here since I always check it prior to passing it through but OK
        self.__check_valid_idx(location)
        data = self.data
        return data[self.__idxs_tuple(location)]

    @property
    def now(self):
        
        # returns an n-dimensional numpy array of all the data at the current time

        current_time = self.domain.time
        time_axis = self.domain.time_axis
        return self.get_data({time_axis:current_time})

    def set_data(self,data,location,allow_override):

        # combines self's data at the location with the input data (merging unknown and known values)
        # then sets self's data to merged data


        self.__check_valid_idx(location)
        
        existing_data = self.data[self.__idxs_tuple(location)] 



        # this approach is a much faster alternative to checking whether each value is an Unknown object
        # faster since this is done in parallel while those checks (even with vectorize) wouldn't


    
        if self.domain.check_bc:

            def find_unknowns(array):
                # any number times 0 would be 0, so any number --> False
                # an unknown times 0 returns itself, so Unknown --> True
                # this check is done in parallel over entire, so much faster
                return array*0!=0
            

            unknown_mask = find_unknowns(data)
            known_mask = np.invert(unknown_mask)


            existing_unknowns = find_unknowns(existing_data[known_mask])
            overriding = not np.all(existing_unknowns)

            if allow_override and overriding:
                # override allowed when bc overrides initial conditions or vice versa (initial setup)
                warnings.warn("Overriding values",category=Warning)
            elif not allow_override and overriding:
                raise Exception("attempting to override values")


        else:

            unknown_mask = np.ma.masked_invalid(data).mask

            #unknown_mask = data==np.nan


        data[unknown_mask] = existing_data[unknown_mask]


        self.data[self.__idxs_tuple(location)]  = data

    def __check_valid_idx(self,idxs):

        # checks if idx is in the form of a dictionary like {"a":1,"b":2} where they're all axis less than their lengths
        
        assert type(idxs)==dict, "indexing must be a dictionary"

        axes_lengths = dict(zip(self.domain.axes_names,self.domain.axes_lengths))

        for axis in idxs:
            assert type(axis)==str, f"{axis} is not a string"
            assert axis in self.domain.axes_names, f"{axis} is not an axis in the domain"
            assert axes_lengths[axis]>idxs[axis], f"{axis} goes out of bounds, it has length {axes_lengths[axis]}"

    def __idxs_tuple(self,idxs):

        # converts a dictionary of idxs to a tuple of indices which can be used to index the numpy array
        # {a:1,b:2} --> (1,2,:)            if axes are a,b,c

        # TODO might make more sense to make this method of domain instead
        idxs_filled = dict()
        for axis in self.domain.axes:
            if axis not in idxs.keys():
                idxs_filled[axis] = slice(None)
            else:
                idxs_filled[axis] = idxs[axis]

        return tuple(idxs_filled[axis] for axis in self.domain.axes)

    def __str__(self) -> str:
        return f"{len(self.domain.axes_names)}-dimensional Field, dimension lengths: {dict(zip(self.domain.axes_names,self.domain.axes_lengths))}"
       
    def __field_op(op1,op2,op):

        def get_operand_data(op):
            # both extracts operand data and adds the domains for later comparison

            if isinstance(op,Field):
                domains.append(op.domain)
                return op.data
            elif type(op)==float or type(op)==int:
                return op
            else:
                raise ValueError("can only perform arithmetic between fields or between fields and numbers")


        domains = []


        if op2==None:
            # case for single argument, i think just negation
            # TODO update this
            new_field = Field(op1.dims,None)
            new_field.data = [op(val) for val in op1.data]

        new_field = Field(op1.domain)
        new_field.data = op(get_operand_data(op1),get_operand_data(op2))


        if len(domains)==2 and domains[0]!=domains[1]:
            raise ValueError("can only perform arithmetic operations between fields with the same domain")
        

        return new_field
       
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
        raise Exception("cannot divide by field")
    
    def __rpow__(self,other):
        raise Exception("cannot raise to field")
    

