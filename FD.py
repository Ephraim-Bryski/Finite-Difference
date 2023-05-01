import numpy as np
from tabulate import tabulate
import operator
import math
import numexpr
import itertools




# TODO replace with class I wrote on scratch



class Domain:
    def __init__(self,axes):
        # dimensions is a dictionary with the dimension object and range
        # TODO: stricter assertion about what's inside
        # TODO: split into space and time?
        assert type(axes)==dict, "axes must be input dictionary of dimensions and range of values"


        for i in range(len(axes)):
            axis_name = list(axes.keys())[i]
            axis_values = list(axes.values())[i]
            assert type(axis_name)==str, "axes keys must be strings, the name of the axis"
            assert type(axis_values)==range, "axes values must be ranges"

            axes[axis_name] = list(axis_values) 


        self.axes = axes
        

        # would also have information like periodicity


    @property
    def axes_lengths(self):
        return [len(list(axis_range)) for axis_range in self.axes.values()]

    @property
    def axes_names(self):
        return list(self.axes.keys())
    
    @property
    def axes_values(self):
        ranges = self.axes.values()
        return [list(axis_range) for axis_range in ranges]
    


class Field:


    def __init__(self,domain):
        
        assert isinstance(domain,Domain), "domain must be a Domain object"

        self.domain = domain
        axes_lengths = [len(value) for value in domain.axes.values()]
        data_length = math.prod(axes_lengths)
        #self.data = [None for _ in range(data_length)]

        self.data = np.full((data_length,1),np.nan)  # converting the data to a numpy array for rapid elementwise arithmetic


    # TODO: has not been modified since I switched to making Fields take Domains
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


            row_counts = [ field.dims[row_dim] for row_dim in row_dims]
            n_rows = int(np.product(row_counts)) # if row_counts is empty (single row) product becomes 1.0 --> convert to 1


            col_counts = [ field.dims[col_dim] for col_dim in col_dims]
            n_cols = int(np.product(col_counts)) # if col_counts is empty (single column) product becomes 1.0 --> convert to 1



            M =np.matrix([[None for j in range(n_cols)] for i in range(n_rows)])


            for i in range(n_rows):
                for j in range(n_cols):
                    row_idxs = Field.__get_dim_idxs(row_counts,i)
                    col_idxs = Field.__get_dim_idxs(col_counts,j)

                    idxs = {**dict(zip(row_dims,row_idxs)),**dict(zip(col_dims,col_idxs))}
                    field_idx = self.__get_field_idx(idxs)
                    M_val = field.data[field_idx]
                    M[i,j] = M_val


            row_placements = Field.__cum_placement(row_counts)
            col_placements = Field.__cum_placement(col_counts)

            return {"rows":dict(zip(row_dims,row_placements)),\
                    "cols":dict(zip(col_dims,col_placements)),\
                    "data":M}

        def empty_np(n_rows,n_cols):
            nested_list = [[None for i in range(n_cols)] for j in range(n_rows)]
            return np.array(nested_list)

        def get_matrix_idxs(n_elem,dims):
            # dims is a dictionary specifying the cumulative product for each dims
            def get_dim_idxs(n_idxs,inner,outer,dim_name):
                # inner: 2, outer 8: means:
                # 001122330011223300112233...

                def strict_int(val):
                    assert int(val)==val,"not even number??"
                    return int(val)   


                n_per_cycle = strict_int(outer/inner)
                n_cycles = strict_int(n_idxs/outer)

                

                idxs = [dim_name]

                for i in range(n_cycles):
                    for j in range(n_per_cycle):
                        for k in range(inner):
                            idxs.append(j)

                return idxs

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

                idxs = get_dim_idxs(n_elem,inner,outer,dim_names[i])
                nested_idxs.append(idxs)
            
            return np.array(nested_idxs)




        dims = list(self.dims.keys())
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
            assert set(rows+cols)==set(dims), "rows and columns have to combine to field dimensions"
        elif "rows" not in keys and "cols" not in keys:
            rows = dims
            cols = []


        matrix = construct_matrix(self,rows,cols)

        row_dims = matrix["rows"]
        col_dims = matrix["cols"]
        data = matrix["data"]

        n_rows = data.shape[0]
        n_cols = data.shape[1]

        n_row_dims = len(row_dims)
        n_col_dims = len(col_dims)


        padded_data = empty_np(1+n_rows,1+n_cols)
        padded_data[1:,1:] = data


        row_idxs = np.transpose(get_matrix_idxs(n_rows,row_dims))
        col_idxs = get_matrix_idxs(n_cols,col_dims)

        top_left_blank = empty_np(n_col_dims,n_row_dims)

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
    def __dict_agree(dict1,dict2):
        # check if all values in dict2 are inside and agree with dict1
        for i in range(len(dict2)):
            key = list(dict2.keys())[i]
            val = list(dict2.values())[i]

            if key not in list(dict1.keys()):
                return False
            if dict1[key]!=val:
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

    def __get_value(self,idxs):
        idx = self.__get_field_idx(idxs)
        return self.data[idx]

    def __set_value(self,idxs,val):
        idx = self.__get_field_idx(idxs)
        self.data[idx] = val


    @staticmethod
    def __check_slice_loc(field_dims,slice_loc):
        # checks if dict1>dict2 for all keys in dict2
        # used for checking if slice goes out of field
        for key in slice_loc.keys():
            if field_dims[key]<=slice_loc[key]:
                return False
        return True
    

    def set(self,expression,location={}):
        

        def transpose(nested_list):
            transposed_list = [[row[i] for row in nested_list] for i in range(len(nested_list[0]))]
            return transposed_list


        axes_lengths_dict = dict(zip(self.domain.axes_names,self.domain.axes_lengths))
        
        assert Field.__check_slice_loc(axes_lengths_dict,location), "slice location not in bounds of field"




        # TODO: to find dimensions it would have to figure out what variables there are
        # this allows me to use those assertions
        # slice_dims = set(slice.dims.keys())


        
        field_dims = set(self.domain.axes_names)
        loc_dims = set(location.keys())
        # assert field_dims-slice_dims==loc_dims , "dimnsions of location must be dimensions in field but not in slice"
        # assert len(slice_dims-field_dims)==0   , "dimensions in slice not in field"
        # Field.__dict_agree(self.dims,slice.dims), "dimension disagreement with slice"


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
            data = numexpr.evaluate(expression,subs)
        except:
            raise Exception("either variable mismatch or cannot parse expression")


        # if the expression is just a constant, numexpr just returns a single value instead of an array
        if data.shape==():
            n_subs = len(value_combs[0])
            data = np.full((n_subs,1),float(data))



        # need to take location into account
        # iterate through all indexes of slice
        for i in range(len(data)):
            # using get_dim_idxs only works because the way itertools flattens the data is the same way i flatten the field data
            dim_idxs_num = Field.__get_dim_idxs(substitute_axes_lengths,i)
            dim_idxs = dict(zip(substitute_axes_names,dim_idxs_num))

            val = data[i]

            field_idxs = {**dim_idxs,**location}

            self.__set_value(field_idxs,val)

     

    # TODO: not sure what to do with this, could return a Slice or Subfield, which inherits from Tensor but doesnt allow for math operations?
    # or could just set Domain to None, which then prohibits calculations
    def __get_slice(self,loc):

        def cut_dict(dict2,cut_keys):
            # construct a dict with cut_keys removed
            new_dict = {}
            for i in range(len(dict2)):
                key = list(dict2.keys())[i]
                val = list(dict2.values())[i]

                if key not in cut_keys:
                    new_dict[key] = val

            return new_dict

        def make_dict(vals,keys,req_keys):
            # constructs dictionary combining val and keys but only for keys in req_keys
            cons_dict = {}
            for i in range(len(keys)):
                key = keys[i]
                val = vals[i]
                if key in req_keys:
                    cons_dict[key] = val
            return cons_dict
        

        field_dims = set(self.dims.keys())
        loc_dims = set(loc.keys())

        assert Field.__check_slice_loc(self.dims,loc), "location out of range of field"
        assert loc_dims.issubset(field_dims), "all indexed dimension must be field dimensions" 

        if field_dims==loc_dims:
            # everything is indexed, so return a number
            return self.__get_value(loc)
        

        # function to get slice of field, returns field with lower dimensionality



        dims = self.dims
        data = self.data


        slice_dims = cut_dict(dims,list(loc.keys()))
        slice = Field(slice_dims,None)
        

        dim_names = list(dims.keys())
        dim_lengths = list(dims.values())

        


        for i in range(len(data)):
            val = data[i]
            dim_idxs = Field.__get_dim_idxs(dim_lengths,i)
            if Field.__dict_agree(dict(zip(dim_names,dim_idxs)),loc):
                slice_idxs = make_dict(dim_idxs,dim_names,slice_dims)
                slice.__set_value(slice_idxs,val)
        return slice


    def __str__(self) -> str:
        return f"{len(self.dims)}-dimensional Field, dimension lengths: {self.dims}"
       
    
    # TODO instead just check for whether the domains match


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

    # this method was used when there was no 
    @staticmethod
    def __OLD_field_op(op1,op2,op):


        # WONT USE
        def dicts_equiv(d1,d2):
            if set(d1)!=set(d2):
                return False
            
            for key in d1.keys():
                if d1[key]!=d2[key]:
                    return False
                
            return True

        # still allow you to add numbers to field
        def field_num_op(field,number):
            # for performing operation between a field and something else
            new_field = Field(field.dims,None)
            assert type(number)==float or type(number)==int, "can only perform arithmetic operations with fields and numbers"

            new_data = [op(val,number) for val in field.data]
            new_field.data = new_data
            return new_field
        
        op1_field = isinstance(op1,Field)
        op2_field = isinstance(op2,Field)

        if op2==None:
            # case for single argument, i think just negation
            new_field = Field(op1.dims,None)
            new_field.data = [op(val) for val in op1.data]
            return new_field
        if not op1_field and not op2_field:
            raise Exception("no field arguments? (shouldnt happen)")
        elif op1_field and not op2_field:
            return field_num_op(op1,op2)
        elif op2_field and not op1_field:
            return field_num_op(op2,op1)
        else:
            assert dicts_equiv(op1.dims,op2.dims), "fields must have same dimensions"

            new_field = Field(op1.dims,None)

            data1 = op1.data
            data2 = op2.data
            

            new_data = [op(val1,val2) for val1,val2 in zip(data1,data2)]

            new_field.data = new_data

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
    


    def __map_kernel(self,kernel,shift,loc):
        # NOT USED (probably won't use)

        # almost same as set_slice but allows the kernel to have smaller dimensions than the full slice

        # kernel can just be any field i want to put in a larger field

        # modifies the field (doesnt return anything)

        # add kernel into the field
        # kernel dimensions must be a subset field dimensions
            # intersection --> shift
            # excess --> loc
        # shift is a dictionary of shared dimensions saying how shifted over it is
        # loc is a dictionary of additional dimensions saying where to place the kernel

        def add_dicts(dict1,dict2):
            # used for adding the shift to the kernel indexes
            sum_dict = {}
            for key in dict1.keys():
                sum_dict[key] = dict1[key]+dict2[key]
            return sum_dict
        
        def add_dict_val(dict1,val):
            # used for adding 1 to the lcoation dictionary
            sum_dict = {}
            for key in dict1.keys():
                sum_dict[key] = dict1[key]+val
            return sum_dict
        
        def compare_dicts(dict1,dict2):
            # checks if dict1>=dict2 for all keys
            # used for checking if kernel goes out of field
            sum_dict = {}
            for key in dict1.keys():
                if dict1[key]<dict2[key]:
                    return False
            return True
        
        


        kernel_dims = set(kernel.dims.keys())
        field_dims = set(self.dims.keys())
        shift_dims = set(shift.keys())
        loc_dims = set(loc.keys())


        max_shifted_dims = add_dicts(kernel.dims,shift)
        max_loc_dims =add_dict_val(loc,1)

        max_dims = {**max_shifted_dims,**max_loc_dims}

        assert compare_dicts(self.dims,max_dims), "kernel goes out of bounds of field"

        assert kernel_dims==shift_dims
        assert field_dims-kernel_dims==loc_dims
        assert len(kernel_dims-field_dims)==0


        data = kernel.data

    


        dim_lengths = np.array(list(kernel.dims.values()))
        dim_names = kernel.dims.keys()
        for i in range(len(data)):
            dim_idxs = Field.__get_dim_idxs(dim_lengths,i)
            dim_idxs = dict(zip(dim_names,dim_idxs))

            kernel_idx = kernel.__get_field_idx(dim_idxs)
            kernel_val = kernel.data[kernel_idx]

            field_idxs = {**add_dicts(dim_idxs,shift),**loc}

            self.__set_value(field_idxs,kernel_val)

    # what other public methods:
        # arithmetic operations
        # numerical calculus


