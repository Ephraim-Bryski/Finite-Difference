import numpy as np
from tabulate import tabulate
import operator
import math
import numexpr
import itertools
import warnings
import numbers
from scipy import signal

# example of use: for wave equation deta/dx would be on the edges, then d/dx deta/dx would be back on the squares, bc can be applied to deta/dx on either side before differentiating again



class Domain:
    # nonconstant property with current time? all operations then are only performed at that time (since the Field has access to Domain props)
    def __init__(self,axes,**kwargs):
        # dimensions is a dictionary with the dimension object and range

        def check_evenly_spaced_array(vals):
            try:
                iter(vals)
            except:
                raise TypeError("must be iterable")
            
            if not np.all([isinstance(val,numbers.Number) for val in vals]):
                raise TypeError("all values must be numbers")

            diffs = np.diff(vals)
            tolerance = 10**-9 # numpy linspace isn't perfectly spaced
            if not np.all([abs(diffs[0]-diff)<tolerance for diff in diffs]):
                raise ValueError("all values must be evenly spaced")
            

        keywords = list(kwargs.keys())
        

        assert np.all([keyword in ["periodic","time"] for keyword in keywords]), "keyword must be periodic or time"



        if "periodic" in keywords:
            periodic = list(kwargs["periodic"])
            assert set(periodic).issubset(set(axes)), f"{periodic} not a dimension"

        else:
            periodic = []

        if "time" in keywords:
            time_axis = kwargs["time"]
            assert type(time_axis)==str, "time axis must be a string"
            assert time_axis in axes, "time axis must be one of the axes"
            assert time_axis not in periodic, "time axis cant be periodic"
        else:
            time_axis = None

        assert type(axes)==dict, "axes must be input dictionary of dimensions and range of values"

    

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
        

        # would also have information like periodicity


    def increment_time(self):
        assert self.time_axis!=None, "need time axis to increment time"

        self.time+=1



    @property
    def dt(self):
        return dict(zip(self.axes_names,self.axes_step_size))[self.time_axis]


    @property
    def n_time_steps(self):
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
    def __init__(self,values,center_idx,axis,domain):
            # would then use the axes_step_size property to construct the kernel values
        assert type(values)==list, "values must be a list"
        assert type(center_idx)==int, "central index must be an integer"
        assert axis in domain.axes_names, "dimension must be in the domain"
        assert len(values)>center_idx, "center_idx must be an index within the kernel size"
        
        kernel = dict()
        for i in range(len(values)):
            value = values[i]
            shifted_idx = i-center_idx
            kernel[shifted_idx] = value
        self.values = kernel
        self.axis = axis

  
class Unknown:
    def __init__(self):
        pass

    def __repr__(self):
        return "?"

class Field:

    @property
    def current(self):
        current_time = self.domain.time
        time_axis = self.domain.time_axis

        return self.get_data({current_time:time_axis})

    def __set_data(self,data,location,allow_override):

        # combines self's data at the location with the input data (merging unknown and known values)
        # then sets self's data to merged data


        self.__check_valid_idx(location)


        existing_data = self.data[self.__idxs_tuple(location)] 


        def merge_values(v_new,v_existing):

            new_known = not isinstance(v_new,Unknown)
            exi_known = not isinstance(v_existing,Unknown)

            if new_known and exi_known:
                # when you apply boundary conditions, you may override initial condition data and vice versa
                # nothing to do about that
                if allow_override:
                    warnings.warn("Overriding values",category=Warning)
                    return v_new
                else:
                        raise Exception("attempting to override values")

            elif new_known and not exi_known:
                return v_new
            elif not new_known and exi_known:
                return v_existing
            else:
                return v_new # could also do v_existing or Unknown()
            
        self.data[self.__idxs_tuple(location)]  = np.vectorize(merge_values,otypes=["object"])(data,existing_data)


    def __check_valid_idx(self,idxs):
        # checks if idx is in the form of a dictionary like {"a":1,"b":2} where they're all axis less than their lengths
        
        assert type(idxs)==dict, "indexing must be a dictionary"


        axes_lengths = dict(zip(self.domain.axes_names,self.domain.axes_lengths))

        for axis in idxs:
            assert type(axis)==str, f"{axis} is not a string"
            assert axis in self.domain.axes_names, f"{axis} is not an axis in the domain"
            assert axes_lengths[axis]>idxs[axis], f"{axis} goes out of bounds, it has length {axes_lengths[axis]}"




    def set_der(self,f,kernel):
        # updates self by taking the derivative of f using the kernel
        # only performs the calculations at the current time

        assert isinstance(f,Field)
        assert isinstance(kernel,Kernel)


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

        diff = np.zeros(current_data.shape,dtype="object")

        for i in range(len(kernel.values)):
            diff+=weights[i]*np.roll(current_data,shifts[i],axis=axis_n)

        left_cut = -shifts[0]
        right_cut = shifts[-1]


        start_idxs = [slice(None)]*n_axes
        start_idxs[axis_n] = slice(0,left_cut)

        end_idxs = [slice(None)]*n_axes
        end_idxs[axis_n] = slice(kernel_axis_len-right_cut,kernel_axis_len)

        periodic = kernel.axis in self.domain.periodic
        # instead of cutting it replace the extra ones with Unknowns
        if not periodic:
            
            diff[tuple(start_idxs)] = Unknown()
            diff[tuple(end_idxs)] = Unknown()






        self.__set_data(diff,current_time,allow_override=False)
        
        
        


        return



    def time_step(self,f_prime):


        assert isinstance(f_prime,Field)

        

        dt = self.domain.dt 
        time_axis = self.domain.time_axis

        if time_axis==None:
            raise Exception("cant time step since there's no time axis")

        current_time = self.domain.time
        new_time = current_time+1

        f_prime_current = f_prime.get_data({time_axis:current_time})
        f_current = self.get_data({time_axis:current_time})


        # right now just assuming euler timestep
        f_new = f_current + f_prime_current*dt


        self.__set_data(f_new,{time_axis:new_time},allow_override=False)




    def __init__(self,domain):
        
        assert isinstance(domain,Domain), "domain must be a Domain object"

        self.domain = domain

        self.data = np.full(self.domain.axes_lengths,Unknown(),dtype="object")
        #axes_lengths = [len(value) for value in domain.axes.values()]
        #self.data = [None for _ in range(data_length)]



        #self.data = np.full((data_length,1),)  # converting the data to a numpy array for rapid elementwise arithmetic


    def __idxs_tuple(self,idxs):
        # TODO might make more sense to make this method of domain instead
        idxs_filled = dict()
        for axis in self.domain.axes:
            if axis not in idxs.keys():
                idxs_filled[axis] = slice(None)
            else:
                idxs_filled[axis] = idxs[axis]


        return tuple(idxs_filled[axis] for axis in self.domain.axes)



    def set_expression(self,expression,location={}):

        self.__check_valid_idx(location)


        def transpose(nested_list):
            transposed_list = [[row[i] for row in nested_list] for i in range(len(nested_list[0]))]
            return transposed_list

        # evaluate the expression for all combinations of dimension values

        # only evaluate the expression over dimensions not in location:

        domain_axes_names = self.domain.axes_names
        domain_axes_values = self.domain.axes_values
        domain_axes_lengths = self.domain.axes_lengths

        substitute_axes_names = []
        substitute_axes_values = []
        substitute_axes_lengths = []



        for i in range(len(domain_axes_names)):
            axis_name = domain_axes_names[i]
            axis_values = domain_axes_values[i]
            axis_lengths = domain_axes_lengths[i]

            if axis_name not in location.keys():
                substitute_axes_names.append(axis_name)
                substitute_axes_values.append(axis_values)
                substitute_axes_lengths.append(axis_lengths)

        # constructs nested list of all combinations of values
        value_combs = transpose([list(comb) for comb in itertools.product(*substitute_axes_values)])
        subs = dict(zip(substitute_axes_names,value_combs))

        try:
            data_flat = numexpr.evaluate(expression,subs)
        except:
            raise Exception("either variable mismatch or cannot parse expression")


        # if the expression is just a constant, numexpr just returns a single value instead of an array
        if data_flat.shape==():
            n_subs = len(value_combs[0])
            data_flat = np.full((n_subs,1),float(data_flat))
        
        data = np.reshape(data_flat,substitute_axes_lengths)

        self.__set_data(data,location,allow_override=True)



    def get_data(self,location={}):
        

        self.__check_valid_idx(location)
        
        data = self.data

        return data[self.__idxs_tuple(location)]






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
    

