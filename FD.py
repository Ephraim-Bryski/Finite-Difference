import numpy as np
from tabulate import tabulate
import operator
import math
import numexpr
import itertools


# TODO option for kernels where output is in the middle of the input e.g fi+1/2=(fi+1-fi)/h   -->   would create new field one row larger (whole to half numbers, or squares to edges) or one row smaller (half to whole numbers, or edges to squares)
# example of use: for wave equation deta/dx would be on the edges, then d/dx deta/dx would be back on the squares, bc can be applied to deta/dx on either side before differentiating again

# TODO kind of sad cause it was a huge waste of time, but I should just construct the fields using numpy arrays instead, would be much simpler


class Domain:
    # nonconstant property with current time? all operations then are only performed at that time (since the Field has access to Domain props)
    def __init__(self,axes,**kwargs):
        # dimensions is a dictionary with the dimension object and range
        # TODO: split into space and time?

        def check_evenly_spaced_array(vals):
            try:
                iter(vals)
            except:
                raise TypeError("must be iterable")
            
            if not np.all([type(val)==float or type(val)==int for val in vals]):
                raise TypeError("all values must be numbers")

            diffs = np.diff(vals)
            if not np.all([diffs[0]==diff for diff in diffs]):
                raise ValueError("all values must be evenly spaced")
            

        keywords = list(kwargs.keys())

        if keywords==[]:
            periodic = []
        elif keywords==["periodic"]:
            periodic = list(kwargs.values())[0]
            assert type(periodic)==list, "periodic must be list of dimensions"
            assert set(periodic).issubset(set(axes)), f"{periodic} not a dimension"
        else:
            invalid_keywords = [keyword for keyword in keywords if keyword!="periodic"]
            raise ValueError(f"invalid keyword arguments: {invalid_keywords}")





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
        

        # would also have information like periodicity


    def step_time(self,time_axis):
        # TODO: allow for different times of timestep (runge-kutta)
        # for now just doing euler timestep
        assert type(time_axis)==str, "time_axis must be name of axis"
        assert time_axis in self.axes_names, "time_axis must be name of axis"

        self.time+=1




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
    def __init__(self,values,center_idx,dimension,domain):
        # TODO: construct it based on derivative order and approx order instead
            # would then use the axes_step_size property to construct the kernel values
        assert type(values)==list, "values must be a list"
        assert type(center_idx)==int, "central index must be an integer"
        assert dimension in domain.axes_names, "dimension must be in the domain"
        assert len(values)>center_idx, "center_idx must be an index within the kernel size"
        kernel = dict()
        for i in range(len(values)):
            value = values[i]
            shifted_idx = i-center_idx
            kernel[shifted_idx] = value
        self.values = kernel
        self.dimension = dimension

    def kernel(der_order,approx_order,type,step):
        # not sure if im even gonna use this
        possible_types = ["forward","backward","central"]
    
        assert type in possible_types, f"types must be one of {possible_types}"

        combination = (der_order,approx_order,type)

        if combination == (1,1,"forward"):
            kernel = {0:-1,1:1}
        elif combination == (1,2,"central"):
            kernel = {-1:-0.5,0:0,1:0.5}
        elif combination == (1,1,"backward"):
            kernel =  {-1:-1,0:1}
        elif combination == (2,1,"central"):
            kernel = {-2:-1,0:0,1:1}
        else:
            raise Exception("not valid or haven't implemented")

        kernel = {k: kernel[k]/step**der_order for k in kernel}
        return kernel



class Unknown:
    def __init__(self):
        pass

    @staticmethod
    def __multiply(value):
        if value==0:
            return 0
        else:
            return Unknown()
                
    def __mul__(self,other):
        return Unknown.__multiply(other)

    def __rmul__(self,other):
        return Unknown.__multiply(other)

    def __add__(self,_):
        return Unknown()
    
    def __radd__(self,_):
        return Unknown()
    
    def __div__(self,_):
        return Unknown()
    
    def __rdiv__(self,_):
        return Unknown()
    
    def __pow__(self,_):
        return Unknown()
    
    def __rpow__(self,_):
        return Unknown()

    def __repr__(self):
        return "?"    


class Field:

    # TODO would probably want kernel to be constructed in diff instead of the user making it
    def diff(self,kernel):
        assert isinstance(kernel,Kernel),"input must be a kernel object"


        data_flat = self.data.flatten()

        data_length = len(data_flat)
        dim_names = self.domain.axes_names
        dim_lengths = self.domain.axes_lengths
        kernel_vals = kernel.values

        dim_length = dict(zip(dim_names,dim_lengths))[kernel.dimension]


        kernel_min_idx = list(kernel_vals.keys())[0]
        kernel_max_idx = list(kernel_vals.keys())[-1]
    

        diff_matrix = np.zeros((data_length,data_length),dtype=object)

        periodic = kernel.dimension in self.domain.periodic


        def periodic_idx(idx,length):
            if idx<0:
                return idx+length
            else:
                return idx%length
            

        for row_idx in range(data_length):
            dim_idxs_vals = Field.__get_dim_idxs(dim_lengths,row_idx)
            dim_idxs = dict(zip(dim_names,dim_idxs_vals))

            dim_idx = dim_idxs[kernel.dimension]

            out_of_bounds = dim_idx+kernel_min_idx<0 or dim_idx+kernel_max_idx>=dim_length
            


            if out_of_bounds and not periodic:
                diff_matrix[row_idx,:] = Unknown()
                continue
            for kernel_idx in kernel_vals:
                val = kernel_vals[kernel_idx]

                # shifted_dim_idxs = dim_idxs.copy()
                # shifted_dim_idxs[kernel.dimension]+=kernel_idx

                shifted_dim_idxs = dim_idxs.copy()
                kernel_dim_idx = shifted_dim_idxs[kernel.dimension]

                shifted_kernel_dim_idx = kernel_dim_idx+kernel_idx

                shifted_dim_idxs[kernel.dimension] = periodic_idx(shifted_kernel_dim_idx,dim_length)

                col_idx = self.__get_field_idx(shifted_dim_idxs)      
                diff_matrix[row_idx,col_idx] = val

        new_data_flat = diff_matrix @ data_flat
        new_data = np.reshape(new_data_flat,self.domain.axes_lengths)
        new_field = Field(self.domain)
        new_field.data = new_data
        return new_field




    def __init__(self,domain):
        
        assert isinstance(domain,Domain), "domain must be a Domain object"

        self.domain = domain

        self.data = np.full(self.domain.axes_lengths,Unknown(),dtype="object")
        #axes_lengths = [len(value) for value in domain.axes.values()]
        #self.data = [None for _ in range(data_length)]



        #self.data = np.full((data_length,1),)  # converting the data to a numpy array for rapid elementwise arithmetic


    # I might just not use show anymore though, and instead rely on plotting
    def show(self,**layout):
        # not using __str__ since that doesn't allow for custom arguments
        # instead called with my_field.show()

        # prints out field flattend out into a matrix (flattens, does NOT take a slice)
        # row_dims and col_dims are lists of dimension names
        def construct_matrix(field,row_dims,col_dims):

            # flattens the field to a 2D matrix so you can see it
            # row_dims and col_dims are list of dims names
            # order dims are listed specificies how it's arranged
            # e.g row_dims = ["a","b"]  -->  a1b1,a1b2,a2b1,a2b2

            axes = dict(zip(field.domain.axes_names,field.domain.axes_lengths))

            row_counts = [axes[row_dim] for row_dim in row_dims]
            n_rows = int(np.product(row_counts)) # if row_counts is empty (single row) product becomes 1.0 --> convert to 1


            col_counts = [axes[col_dim] for col_dim in col_dims]
            n_cols = int(np.product(col_counts)) # if col_counts is empty (single column) product becomes 1.0 --> convert to 1



            M =np.matrix([[None for j in range(n_cols)] for i in range(n_rows)])

            flat_data = field.data.flatten()
            for i in range(n_rows):
                for j in range(n_cols):
                    row_idxs = Field.__get_dim_idxs(row_counts,i)
                    col_idxs = Field.__get_dim_idxs(col_counts,j)
                    

                    idxs = {**dict(zip(row_dims,row_idxs)),**dict(zip(col_dims,col_idxs))}
                    field_idx = self.__get_field_idx(idxs)
                    M_val = flat_data[field_idx]
                    M[i,j] = M_val


            row_placements = Field.__cum_placement(row_counts)
            col_placements = Field.__cum_placement(col_counts)

            return {"rows":dict(zip(row_dims,row_placements)),\
                    "cols":dict(zip(col_dims,col_placements)),\
                    "data":M}


        def get_matrix_dim_values(n_elem,dims):
            # dims is a dictionary specifying the cumulative product for each dims
            def get_dim_values(n_idxs,inner,outer,dim_name):
                # inner: 2, outer 8: means:
                # 001122330011223300112233...

                def strict_int(val):
                    assert int(val)==val,"not even number??"
                    return int(val)   


                n_per_cycle = strict_int(outer/inner)
                n_cycles = strict_int(n_idxs/outer)

                

                dim_values = [dim_name] # starting off with a label for the matrix display

                axes_values = dict(zip(self.domain.axes_names,self.domain.axes_values))

                for i in range(n_cycles):
                    for j in range(n_per_cycle):
                        for k in range(inner):
                            #idxs.append(j)
                            dim_values.append(axes_values[dim_name][j])

                return dim_values
            dim_names = list(dims.keys())
            dim_cum = np.array(list(dims.values()))
            n_dims = len(dim_names)
            nested_idxs = []
            for i in range(n_dims):
                inner = dim_cum[i]
                
                if i==0:
                    outer = n_elem
                else:
                    outer = dim_cum[i-1]

                idxs = get_dim_values(n_elem,inner,outer,dim_names[i])
                nested_idxs.append(idxs)
            
            return np.array(nested_idxs)




        dims = self.domain.axes_names
        keys = list(layout.keys())

        def list_subtract(list1,list2):
            # list1-list2, all elements in list1 and not list2
            return [el for el in list1 if el not in list2]
        
        assert set(keys).issubset({"rows","cols"}), "unknown keyword argument"

        arguments = list(layout.values())
        for argument in arguments:
            assert type(argument)==list, "all layout dimensions must be lists of dimension names"
        

        # dims = ["a","b","c"]
        if "rows" in keys and "cols" not in keys:
            rows = layout["rows"]
            assert set(rows).issubset(set(dims)), "all rows must be field dimensions"
            cols = list_subtract(dims,rows)
        if "cols" in keys and "rows" not in keys:
            cols = layout["cols"]
            assert set(cols).issubset(set(dims)), "all columns must be field dimensions"
            rows = list_subtract(dims,cols)
        if "cols" in keys and "rows" in keys:
            rows = layout["rows"]
            cols = layout["cols"]
            assert set(rows).intersection(set(cols))==set(), "rows and columns cannot share field dimensions"
            assert set(rows+cols)==set(dims), "rows and columns have to combine to field dimensions"
        elif "rows" not in keys and "cols" not in keys:
            rows = []
            cols = dims


        matrix = construct_matrix(self,rows,cols)

        row_dims = matrix["rows"]
        col_dims = matrix["cols"]
        data = matrix["data"]

        n_rows = data.shape[0]
        n_cols = data.shape[1]

        n_row_dims = len(row_dims)
        n_col_dims = len(col_dims)

        padded_data = np.full((1+n_rows,1+n_cols),None)
        padded_data[1:,1:] = data


        row_idxs = np.transpose(get_matrix_dim_values(n_rows,row_dims))
        col_idxs = get_matrix_dim_values(n_cols,col_dims)

        top_left_blank = np.full((n_col_dims,n_row_dims),None)

        if len(col_idxs)==0:
            data_with_rows = np.concatenate((row_idxs,padded_data),1)
            matrix_display = data_with_rows
        elif len(row_idxs)==0:
            data_with_cols = np.concatenate((col_idxs,padded_data),0)
            matrix_display = data_with_cols
        else:
            data_with_rows = np.concatenate((row_idxs,padded_data),1)
            column_header = np.concatenate((top_left_blank,col_idxs),1)
            matrix_display = np.concatenate((column_header,data_with_rows),0)

        print(tabulate(matrix_display))

    @staticmethod
    def __cum_placement(dim_lengths):
        # returns list of how many step each dimension jumps over (the final element is always 1)
        dim_lengths_shifted = np.delete(np.concatenate((dim_lengths,np.array([1]))),0)
        return np.flip(np.cumprod(np.flip(dim_lengths_shifted)))

    @staticmethod
    def __get_dim_idxs(dim_lengths,idx):
        # takes a numpy array of the length of each dimension (order indicates how data is ordered)
        # converts the overall index to an array of indexes for each dimension
        # used:
            # construct_mat, dims are the dims for row or column
            # mapping, need to index all data in field, dims are all dims of field

        assert idx<np.product(dim_lengths)

        placements = Field.__cum_placement(dim_lengths)

        return np.mod(np.floor(idx/placements),dim_lengths).astype(int)

 
    @staticmethod
    def __check_slice_loc(field_dims,slice_loc):
        # checks if dict1>dict2 for all keys in dict2
        # used for checking if slice goes out of field
        for key in slice_loc.keys():
            if field_dims[key]<=slice_loc[key]:
                return False
        return True
    

    def __get_field_idx(self,idxs):
        # SHOULD ONLY BE CALLED PRIVATELY (should never be made public)

        # returns the 1d field index
        # gets the value of a field at a specified index
        
        # idxs must be a dictionary with the keys being the dims and the value being the index


        assert set(self.domain.axes_names)==set(idxs.keys()), "All dimensions of indexing must match dimensions of field"

        idxs_sorted = {key: idxs[key] for key in self.domain.axes_names}
        idx_values = list(idxs_sorted.values())

        dims_length = self.domain.axes_lengths
        place_count = Field.__cum_placement(dims_length)



        assert idx_values>=[0]*len(idx_values), "negative index"
        assert dims_length>idx_values, "index out of range"


        def dot_product(list1,list2):
            product_terms = [a * b for a, b in zip(list1, list2)]
            return sum(product_terms)

        idx_glob = dot_product(place_count,idx_values)

        return idx_glob


    def __idxs_tuple(self,idxs):

        idxs_filled = dict()

        for axis in self.domain.axes:
            if axis not in idxs.keys():
                idxs_filled[axis] = slice(None)
            else:
                idxs_filled[axis] = idxs[axis]


        return tuple(idxs_filled[axis] for axis in self.domain.axes)



    def set(self,expression,location={}):
        

        def transpose(nested_list):
            transposed_list = [[row[i] for row in nested_list] for i in range(len(nested_list[0]))]
            return transposed_list


        axes_lengths_dict = dict(zip(self.domain.axes_names,self.domain.axes_lengths))
        
        assert Field.__check_slice_loc(axes_lengths_dict,location), "slice location not in bounds of field"




        # TODO: do more granular check, first checking if all variables are allowed (might require sympy), this allows for these assertions

        def dict_agree(dict1,dict2):
            # check if all values in dict2 are inside and agree with dict1
            for i in range(len(dict2)):
                key = list(dict2.keys())[i]
                val = list(dict2.values())[i]

                if key not in list(dict1.keys()):
                    return False
                if dict1[key]!=val:
                    return False
            return True

        field_dims = set(self.domain.axes_names)
        loc_dims = set(location.keys())
        # assert field_dims-slice_dims==loc_dims , "dimnsions of location must be dimensions in field but not in slice"
        # assert len(slice_dims-field_dims)==0   , "dimensions in slice not in field"
        # dict_agree(self.dims,slice.dims), "dimension disagreement with slice"


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
            data = np.full((n_subs,1),float(data_flat))
        else:

            data = np.reshape(data_flat,self.domain.axes_lengths)

        self.data[self.__idxs_tuple(location)] = data

        # need to take location into account
        # iterate through all indexes of slice
        """
                for i in range(len(data)):
            # using get_dim_idxs only works because the way itertools flattens the data is the same way i flatten the field data
            dim_idxs_num = Field.__get_dim_idxs(substitute_axes_lengths,i)
            dim_idxs = dict(zip(substitute_axes_names,dim_idxs_num))

            val = data[i]

            field_idxs = {**dim_idxs,**location}


            idx = self.__get_field_idx(field_idxs)
            self.data[idx] = val

        """


    # TODO: function that adds boundary conditions
        # on each time step, you first assign field to something, then apply boundary conditions
        # a bit weird, but easy

    # TODO: right now loc is the indexes, might want to have it the location instead
    def get_data(self,loc={}):
        
        
        assert type(loc)==dict, "must input dictionary"
        # TODO: write private nonstatic method that checks if it's a dictionary with all keys in dimension and numeric values, as this is done in multiple places

        field_dims = set(self.domain.axes_names)
        loc_dims = set(loc.keys())
        dims = dict(zip(self.domain.axes_names,self.domain.axes_lengths))
        data = self.data
        assert loc_dims.issubset(field_dims), "all indexed dimension must be field dimensions" 
        assert Field.__check_slice_loc(dims,loc), "location out of range of field"

        return data[self.__idxs_tuple(loc)]


    """
     def cut_dict(dict2,cut_keys):
            # construct a dict with cut_keys removed
            new_dict = {}
            for i in range(len(dict2)):
                key = list(dict2.keys())[i]
                val = list(dict2.values())[i]

                if key not in cut_keys:
                    new_dict[key] = val

            return new_dict


        slice_dims = cut_dict(dims,list(loc.keys()))



        array = np.zeros(tuple(slice_dims.values()))


       
                for field_idx in range(len(data)):

            
            dim_idxs_vals = Field.__get_dim_idxs(self.domain.axes_lengths,field_idx)
            dim_idxs = dict(zip(self.domain.axes_names,dim_idxs_vals))
            data_in_loc = np.all([dim_idxs[k]==loc[k] for k in loc])

            if not data_in_loc:
                continue

            slice_dim_idxs = tuple(dim_idxs[k] for k in slice_dims)

            
            array[slice_dim_idxs] = data[field_idx]
   

        return array
     """







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
        return Field.__field_op(other,self,operator.add)
    
    def __rsub__(self,other):
        return Field.__field_op(other,self,operator.sub)
    
    def __rmul__(self,other):
        return Field.__field_op(other,self,operator.mul)
    
    def __rtruediv__(self,other):
        raise Exception("cannot divide by field")
    
    def __rpow__(self,other):
        raise Exception("cannot raise to field")
    

