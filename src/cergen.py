"""
Solution for the first homework of the 2023-2024 Spring class of the Introduction to Deep Learning course at METU.

You can find the results of the comparison with numpy at the bottom of the file.

Baran Yancı, 2449015
"""


import math
import numpy as np
import random
import time

from typing import Union


class gergen:
    pass


######################################################################
#-------------------------- HELPER CLASSES --------------------------#
######################################################################
class TupleIndexedList:
    def __init__(self, tuple_indexed_list: list):
        self.tuple_indexed_list = tuple_indexed_list

    def __getitem__(self, index):
        if type(index) == int:
            return self.tuple_indexed_list[index]
        
        if type(index) == tuple:
            val_to_return = self.tuple_indexed_list

            while len(index) > 0:
                val_to_return = val_to_return[index[0]]
                index = index[1:]

            return val_to_return

    def __setitem__(self, index, value):
        if type(index) == int:
            self.tuple_indexed_list[index] = value
            return
        
        if type(index) == tuple:
            val_to_return = self.tuple_indexed_list

            while len(index) > 1:
                val_to_return = val_to_return[index[0]]
                index = index[1:]

            val_to_return[index[0]] = value

    def to_list(self):
        return self.tuple_indexed_list
    

epsilon = 1e-5


class TestResult:
    def __init__(self, test_name: str, durations: list, results: list):
        self.test_name = test_name
        self.durations = durations
        self.results = results
        self.passed = self.did_pass()

    def did_pass(self):
        return np.allclose(self.results[0], self.results[1], epsilon, epsilon)

    def __str__(self):
        str_repr = ""

        str_repr += ("Are the results the same?: " + str(self.passed )+ "\n")

        #Report the time difference
        str_repr += ("Time taken for gergen: " + str(self.durations[0]) + "\n")
        str_repr += ("Time taken for numpy: " + str(self.durations[1]))
        
        return str_repr
    
    def to_obj(self):
        return {
            "test_name": self.test_name,
            "durations": self.durations,
            "passed": self.passed,
            "results": self.results
        }


######################################################################
#------------------------- HELPER FUNCTIONS -------------------------#
######################################################################
def get_total_element_count_from_dimensions(boyut: tuple) -> int:
    total_element_count = 1

    for el in boyut:
        total_element_count *= el

    return total_element_count


def get_total_element_count_from_nested_list(nested_list: list) -> int:
    stringified_nested_list = str(nested_list)

    return stringified_nested_list.count(',') + 1


def create_nested_list(boyut: tuple, aralik_list: list, use_integer: bool) -> list:
    if len(boyut) == 0:
        return random.randint(*aralik_list) if use_integer else random.uniform(*aralik_list)

    total_length = 1

    for el in boyut:
        total_length *= el

    unnested_list = [random.randint(*aralik_list) if use_integer else random.uniform(*aralik_list)
        for _ in range(total_length)
    ]
    
    return nest_list(unnested_list, boyut)


def create_nested_list_with_fill(boyut: tuple, fill) -> list:
    """
    I could've modified the function above to accept another parameter, but I didn't want to interfere with the random number generation
    """
    if len(boyut) == 0:
        return fill

    total_length = 1

    for el in boyut:
        total_length *= el

    unnested_list = [fill for _ in range(total_length)]
    
    return nest_list(unnested_list, boyut)


def get_transpose_of_nested_list(nested_list: list | int | float) -> list | int | float:
    if (
        type(nested_list) == int or
        type(nested_list) == float
    ):
        return nested_list
    
    original_shape = get_dimensions_of_nested_list(nested_list)
    new_shape = original_shape[::-1]

    transposed_nested_list = create_nested_list_with_fill(new_shape, math.inf)

    tuple_indexed_original = TupleIndexedList(nested_list)
    tuple_indexed_transposed = TupleIndexedList(transposed_nested_list)

    """
    this digital clock variable is used to keep track of the indexes. it is a list of 0s with the same length as the original shape, but
    every digit at index i is reset to 0 when the index i is equal to the corresponding dimension of the original shape. this is used to
    keep track of the indexes when iterating over the original nested list.

    let's see an example:

    if the original dimensions are (4, 3, 2, 5), the digital clock will start with value [0, 0, 0, 0]. at every iteration of the loop below
    (which is used to iterate over the original nested list), the digital clock will be incremented by 1. when the first digit of the 
    digital clock is equal to 4, it is reset to 0 and the second digit is incremented by 1. this process continues until the total number
    of iterations is equal to the total number of elements in the original nested list.
    """
    digital_clock = [0 for _ in range(len(original_shape))]

    element_count = get_total_element_count_from_dimensions(original_shape)

    counter = 0

    while True:
        tuple_indexed_transposed[tuple(digital_clock)[::-1]] = tuple_indexed_original[tuple(digital_clock)]

        digital_clock[-1] += 1

        for i in range(len(digital_clock) - 1, -1, -1):
            if digital_clock[i] == original_shape[i]:
                digital_clock[i] = 0
                digital_clock[i - 1] += 1
            else:
                break
        
        counter += 1

        if counter == element_count:
            break

    return tuple_indexed_transposed.to_list()


def get_dimensions_of_nested_list(nested_list: list) -> tuple:
    if type(nested_list) == int or type(nested_list) == float:
        return ()

    boyut_list = []
    current_nested_list = nested_list

    while type(current_nested_list[0]) != int and type(current_nested_list[0]) != float:
        boyut_list.append(len(current_nested_list))
        current_nested_list = current_nested_list[0]

    boyut_list.append(len(current_nested_list))

    return tuple(boyut_list)


def unnest_list(nested_list: list) -> list:
    if (
        type(nested_list) == int or
        type(nested_list) == float
    ):
        return [nested_list]
    
    return [
        el for sublist in nested_list for el in unnest_list(sublist)
    ]


def nest_list(unnested_list: list, boyut: tuple, total_element_count: int = -1) -> list:
    if len(boyut) == 0:
        return unnested_list[0]
    
    if len(boyut) == 1 and boyut[0] == len(unnested_list):
        return unnested_list

    if isinstance(unnested_list, (int, float)):
        return unnested_list

    if total_element_count == -1:
        total_element_count = 1

        for el in boyut[1:]:
            total_element_count *= el

    sublist_element_count = int(total_element_count / boyut[1])

    return [
        nest_list(
            unnested_list[i * total_element_count : (i + 1) * total_element_count], 
            boyut[1:], 
            sublist_element_count
        ) for i in range(boyut[0])
    ]


def map_nested_list(nested_list: list, map_fn) -> list:
    if (
        type(nested_list) == int or
        type(nested_list) == float
    ):
        return map_fn(nested_list)
    
    return [
        map_nested_list(el, map_fn) for el in nested_list
    ]


######################################################################
#------------------------ OPERATION CLASSES -------------------------#
######################################################################

class Operation:
    def __call__(self, *operands):
        """
        Makes an instance of the Operation class callable.
        Stores operands and initializes outputs to None.
        Invokes the forward pass of the operation with given operands.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the forward pass of the operation.
        """
        self.operands = operands
        self.outputs = None
        return self.ileri(*operands)

    def ileri(self, *operands):
        """
        Defines the forward pass of the operation.
        Must be implemented by subclasses to perform the actual operation.

        Parameters:
            *operands: Variable length operand list.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError


"""
The Operation class serves as a base class for operations in the gergen class. It is defined as above.

The call method allows instances of subclasses of Operation to be used as if they were functions, enabling a concise syntax for applying
operations to operands. The ileri method is intended to be overridden by subclasses to define the specific behavior of the operation.
In the context of the gergen class, the Operation class serves as a foundational component for defining various mathematical and tensor
operations. When creating new operations, such as addition, multiplication, or more complex functions, you should define subclasses of
Operation and implement the ileri method to encapsulate the operation’s specific logic. The call method in the Operation base class will
automatically handle the invocation of the ileri method, treating instances of these subclasses as callable objects. To integrate an
operation into the gergen class, instantiate the corresponding Operation subclass and pass the necessary operands (other gergen instances
or scalars) to perform the operation, ultimately returning a new gergen object that represents the result.
"""


class Addition(Operation):
    def adder(self, left: Union[list, int, float], right: Union[list, int, float]) -> Union[list, int, float]:
        """
        For gergen-to-gergen addition, it iterates over corresponding el- ements from both instances, adding them together. If one
        operand is a scalar, this value is added to every element within the gergen instance. The method performs a dimensionality
        check when both operands are gergen instances to ensure their shapes are compatible for element-wise operations. If the
        dimensions do not align, a ValueError is raised, indicating a mismatch in dimensions. Additionally, if the other parameter is
        of an unsupported type, a TypeError is raised to maintain type safety. The outcome of the addition is should be returned in a 
        new gergen object.
        """
        if (
            isinstance(left, (list)) and
            isinstance(right, (list))
        ):
            """
            both gergos (represented by lists)
            """
            
            return ([
                self.adder(
                    left[i],
                    right[i]
                ) for i in range(len(left))
            ])
        
        if (
            isinstance(left, (int, float)) and
            isinstance(right, (int, float))
        ):
            """
            both scalars
            """
            return left + right
        
        if isinstance(left, (int, float)):
            """
            left is scalar
            """
            return ([
                self.adder(left, el) for el in right
            ])

        if isinstance(right, (int, float)):
            """
            right is scalar
            """
            return ([
                self.adder(el, right) for el in left
            ])
        
        raise TypeError('Operands should be of type int, float, or gergen')

    def ileri(self, *operands: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the forward pass of the addition operation.
        Adds the given operands element-wise.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the addition operation.
        """
        if not all(isinstance(operand, (int, float, gergen)) for operand in operands):
            raise TypeError('Operands should be of type int, float, or gergen')

        if len(operands) < 2:
            raise ValueError('Addition operation requires at least two operands')
        
        if len(operands) == 2:
            if (
                isinstance(operands[0], (gergen)) and
                isinstance(operands[1], (gergen))
            ):
                if operands[0].boyut() != operands[1].boyut() and operands[0].boyut() != () and operands[1].boyut() != ():
                    raise ValueError('Operands should have the same shape')

            #! WE WILL NOT USE THE GERGEN OBJECT IN adder FUNCTION. INSTEAD, WE WILL PASS THE listeye() OF THE GERGEN OBJECT.
            neutralised_operands = [
                operand if isinstance(operand, (int, float)) else operand.listeye()
                    for operand in operands
            ]

            #! WE WILL RETURN THE RESULT AS A GERGEN OBJECT.
            return gergen(self.adder(*neutralised_operands))

        result = operands[0]

        result = self(result, operands[1:])

        return result


class Subtraction(Operation):
    def subtractor(self, left: Union[list, int, float], right: Union[list, int, float]) -> Union[list, int, float]:
        """
        This method en- ables element-wise subtraction, either between two gergen instances or between a gergen and a scalar (int/float).
        For gergen-to-gergen subtraction, corresponding elements from each instance are subtracted. When operating with a scalar, the
        scalar value is subtracted from each element of the gergen instance. The method ensures that dimensions are compatible when both
        operands are gergen instances, raising a ValueError if there is a mismatch. If the type of other is not supported, a TypeError is
        raised. The outcome of the subtraction is should be returned in a new gergen ob- ject.
        """
        if (
            isinstance(left, (list)) and
            isinstance(right, (list))
        ):
            """
            both gergos (represented by lists)
            """
            
            return ([
                self.subtractor(
                    left[i],
                    right[i]
                ) for i in range(len(left))
            ])
        
        if (
            isinstance(left, (int, float)) and
            isinstance(right, (int, float))
        ):
            """
            both scalars
            """
            return left - right
        
        if isinstance(left, (int, float)):
            """
            left is scalar
            """
            return ([
                self.subtractor(left, el) for el in right
            ])

        if isinstance(right, (int, float)):
            """
            right is scalar
            """
            return ([
                self.subtractor(el, right) for el in left
            ])
        
        raise TypeError('Operands should be of type int, float, or gergen')

    def ileri(self, *operands: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the forward pass of the addition operation.
        Adds the given operands element-wise.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the addition operation.
        """
        if not all(isinstance(operand, (int, float, gergen)) for operand in operands):
            raise TypeError('Operands should be of type int, float, or gergen')

        if len(operands) < 2:
            raise ValueError('Addition operation requires at least two operands')
        
        if len(operands) == 2:
            if (
                isinstance(operands[0], (gergen)) and
                isinstance(operands[1], (gergen))
            ):
                if operands[0].boyut() != operands[1].boyut() and operands[0].boyut() != () and operands[1].boyut() != ():
                    raise ValueError('Operands should have the same shape')

            #! WE WILL NOT USE THE GERGEN OBJECT IN adder FUNCTION. INSTEAD, WE WILL PASS THE listeye() OF THE GERGEN OBJECT.
            neutralised_operands = [
                operand if isinstance(operand, (int, float)) else operand.listeye()
                    for operand in operands
            ]

            #! WE WILL RETURN THE RESULT AS A GERGEN OBJECT.
            return gergen(self.subtractor(*neutralised_operands))

        result = operands[0]

        result = self(result, operands[1:])

        return result
    

class Multiplication(Operation):
    def multiplier(self, left, right):
        """
        This method fa- cilitates the multiplication of the gergen either with another gergen instance for element-wise multiplication,
        or with a scalar (int/float), yielding a new gergen ob- ject as the result. The other parameter is permitted to be a gergen, an
        integer, or a floating-point number. Error handling is incorporated to manage cases where the other parameter is neither a gergen
        object nor a numerical scalar. If the dimen- sions of two gergen instances do not align for element-wise multiplication, or if an
        incompatible type is provided for other, a TypeError or ValueError is raised.
        """
        if (
            isinstance(left, (list)) and
            isinstance(right, (list))
        ):
            """
            both gergos (represented by lists)
            """
            
            return ([
                self.multiplier(
                    left[i],
                    right[i]
                ) for i in range(len(left))
            ])
        
        if (
            isinstance(left, (int, float)) and
            isinstance(right, (int, float))
        ):
            """
            both scalars
            """
            return left * right
        
        if isinstance(left, (int, float)):
            """
            left is scalar
            """
            return ([
                self.multiplier(left, el) for el in right
            ])

        if isinstance(right, (int, float)):
            """
            right is scalar
            """
            return ([
                self.multiplier(el, right) for el in left
            ])
        
        raise TypeError('Operands should be of type int, float, or gergen')

    def ileri(self, *operands: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the forward pass of the addition operation.
        Adds the given operands element-wise.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the addition operation.
        """
        if not all(isinstance(operand, (int, float, gergen)) for operand in operands):
            raise TypeError('Operands should be of type int, float, or gergen')

        if len(operands) < 2:
            raise ValueError('Multiplication operation requires at least two operands')
        
        if len(operands) == 2:
            if (
                isinstance(operands[0], (gergen)) and
                isinstance(operands[1], (gergen))
            ):
                if operands[0].boyut() != operands[1].boyut() and operands[0].boyut() != () and operands[1].boyut() != ():
                    raise ValueError('Operands should have the same shape')

            #! WE WILL NOT USE THE GERGEN OBJECT IN adder FUNCTION. INSTEAD, WE WILL PASS THE listeye() OF THE GERGEN OBJECT.
            neutralised_operands = [
                operand if isinstance(operand, (int, float)) else operand.listeye()
                    for operand in operands
            ]

            #! WE WILL RETURN THE RESULT AS A GERGEN OBJECT.
            return gergen(self.multiplier(*neutralised_operands))

        result = operands[0]

        result = self(result, operands[1:])

        return result
            

class Division(Operation):
    def divisor(self, left, right):
        """
        This method implements division for the gergen, facilitating element-wise division by a scalar (an integer or a float), and
        encapsulates the result in a new gergen instance. True divi- sion is employed, ensuring that the result is always a floating-point
        number, consistent with Python 3.x division behavior, even if both operands are integers. Error handling mechanism ahould check
        potential issues: if other is zero, a ZeroDivisionError is raised to prevent division by zero. Additionally, if other is not a
        scalar type (int or float), a TypeError is raised to enforce the type requirement for the scalar divisor.
        """
        if right == 0:
            raise ZeroDivisionError('Division by zero is not allowed')

        if (
            isinstance(left, (list)) and
            isinstance(right, (list))
        ):
            """
            both gergos (represented by lists)
            """
            
            return ([
                self.divisor(
                    left[i],
                    right[i]
                ) for i in range(len(left))
            ])
        
        if (
            isinstance(left, (int, float)) and
            isinstance(right, (int, float))
        ):
            """
            both scalars
            """
            return left / right
        
        if isinstance(left, (int, float)):
            """
            left is scalar
            """
            return ([
                self.divisor(left, el) for el in right
            ])

        if isinstance(right, (int, float)):
            """
            right is scalar
            """
            return ([
                self.divisor(el, right) for el in left
            ])
        
        raise TypeError('Operands should be of type int, float, or gergen')

    def ileri(self, *operands: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the forward pass of the addition operation.
        Adds the given operands element-wise.

        Parameters:
            *operands: Variable length operand list.

        Returns:
            The result of the addition operation.
        """
        if not all(isinstance(operand, (int, float, gergen)) for operand in operands):
            raise TypeError('Operands should be of type int, float, or gergen')

        if len(operands) < 2:
            raise ValueError('Multiplication operation requires at least two operands')
        
        if len(operands) == 2:
            if (
                isinstance(operands[0], (gergen)) and
                isinstance(operands[1], (gergen))
            ):
                if operands[0].boyut() != operands[1].boyut() and operands[0].boyut() != () and operands[1].boyut() != ():
                    raise ValueError('Operands should have the same shape')

            #! WE WILL NOT USE THE GERGEN OBJECT IN adder FUNCTION. INSTEAD, WE WILL PASS THE listeye() OF THE GERGEN OBJECT.
            neutralised_operands = [
                operand if isinstance(operand, (int, float)) else operand.listeye()
                    for operand in operands
            ]

            #! WE WILL RETURN THE RESULT AS A GERGEN OBJECT.
            return gergen(self.divisor(*neutralised_operands))

        result = operands[0]

        result = self(result, operands[1:])

        return result


######################################################################
#------------------------- GERGOOOOOOOOOOOO -------------------------#
######################################################################

class gergen:

    __veri = None #A nested list of numbers representing the data
    D = None # Transpose of data
    __boyut = None #Dimensions of the derivative (Shape)

    __adder = Addition()
    __subtractor = Subtraction()
    __multiplier = Multiplication()
    __divider = Division()

    __unnested_veri = None


    def __init__(self, veri=None):
    # The constructor for the 'gergen' class.
    #
    # This method initializes a new instance of a gergen object. The gergen can be
    # initialized with data if provided; otherwise, it defaults to None, representing
    # an empty tensor.
    #
    # Parameters:
    # veri (int/float, list, list of lists, optional): A nested list of numbers that represents the
    # gergen data. The outer list contains rows, and each inner list contains the
    # elements of each row. If 'veri' is None, the tensor is initialized without data.
    #
    # Example:
    # To create a tensor with data, pass a nested list:
    # tensor = gergen([[1, 2, 3], [4, 5, 6]])
    #
    # To create an empty tensor, simply instantiate the class without arguments:
    # empty_tensor = gergen()
        self.__veri = veri
        self.__boyut = get_dimensions_of_nested_list(veri)

    def __getitem__(self, index) -> gergen:
    #Indexing for gergen objects
        if self.__veri is None:
            raise ValueError('Tensor is empty')
        
        tuple_indexed = TupleIndexedList(self.__veri)

        return gergen(tuple_indexed[index])

    def __str__(self):
        #Generates a string representation
        string_to_print = ""

        if self.__veri is None:
            string_to_print += "Boş gergen"

        elif type(self.__veri) == int or type(self.__veri) == float:
            # If the tensor is a scalar, we can directly return the string representation of the scalar.
            string_to_print += "0 boyutlu skaler gergen:\n" + str(self.__veri)

        else:
            # If the tensor is not a scalar, we can make use of __boyut variable to construct a string representation.
            for dim in self.__boyut:
                string_to_print += str(dim) + "x"

            string_to_print = string_to_print[:-1]
            string_to_print += " boyutlu gergen:\n" + str(self.__veri)

        return string_to_print + "\n"


    def __mul__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Multiplication operation for gergen objects.
        Called when a gergen object is multiplied by another, using the '*' operator.
        Could be element-wise multiplication or scalar multiplication, depending on the context.
        """
        return self.__multiplier(self, other)

    def __truediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Division operation for gergen objects.
        Called when a gergen object is divided by another, using the '/' operator.
        The operation is element-wise.
        """
        return self.__divider(self, other)


    def __add__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the addition operation for gergen objects.
        Called when a gergen object is added to another, using the '+' operator.
        The operation is element-wise.
        """
        return self.__adder(self, other)

    def __sub__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Subtraction operation for gergen objects.
        Called when a gergen object is subtracted from another, using the '-' operator.
        The operation is element-wise.
        """
        return self.__subtractor(self, other)

    def __radd__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Defines the addition operation for gergen objects when the left operand is a scalar.
        Called when a scalar is added to a gergen object, using the '+' operator.
        The operation is element-wise.
        """
        return self.__adder(other, self)
    
    def __rsub__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Subtraction operation for gergen objects when the left operand is a scalar.
        Called when a scalar is subtracted from a gergen object, using the '-' operator.
        The operation is element-wise.
        """
        return self.__subtractor(other, self)
    
    def __rmul__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Multiplication operation for gergen objects when the left operand is a scalar.
        Called when a scalar is multiplied by a gergen object, using the '*' operator.
        The operation is element-wise.
        """
        return self.__multiplier(other, self)
    
    def __rtruediv__(self, other: Union['gergen', int, float]) -> 'gergen':
        """
        Division operation for gergen objects when the left operand is a scalar.
        Called when a scalar is divided by a gergen object, using the '/' operator.
        The operation is element-wise.
        """
        return self.__divider(other, self)

    def uzunluk(self):
    # Returns the total number of elements in the gergen
        if type(self.__veri) == int or type(self.__veri) == float:
            return 1
        
        return len(self.duzlestir())
        

    def boyut(self):
    # Returns the shape of the gergen
        return self.__boyut

    def devrik(self):
    # Returns the transpose of gergen
        if self.D is None:
            self.D = gergen(get_transpose_of_nested_list(self.__veri))

        return self.D

    def sin(self):
    #Calculates the sine of each element in the given `gergen`.
        return gergen(map_nested_list(self.__veri, lambda x: math.sin(x)))

    def cos(self):
    #Calculates the cosine of each element in the given `gergen`.
        return gergen(map_nested_list(self.__veri, lambda x: math.cos(x)))

    def tan(self):
    #Calculates the tangent of each element in the given `gergen`.
        return gergen(map_nested_list(self.__veri, lambda x: math.tan(x)))

    def us(self, n: int):
    #Raises each element of the gergen object to the power 'n'. This is an element-wise operation.
        return gergen(map_nested_list(self.__veri, lambda x: x ** n))

    def log(self):
    #Applies the logarithm function to each element of the gergen object, using the base 10.
        return gergen(map_nested_list(self.__veri, lambda x: math.log10(x)))

    def ln(self):
    #Applies the natural logarithm function to each element of the gergen object.
        return gergen(map_nested_list(self.__veri, lambda x: math.log(x)))

    def L1(self):
    # Calculates and returns the L1 norm
        return self.Lp(1)

    def L2(self):
    # Calculates and returns the L2 norm
        return self.Lp(2)

    def Lp(self, p):
    # Calculates and returns the Lp norm, where p should be positive integer
        if p <= 0:
            raise ValueError('p should be a positive integer')

        unnested_list = self.duzlestir()

        return sum([abs(el) ** p for el in unnested_list]) ** (1 / p)

    def listeye(self):
    #Converts the gergen object into a list or a nested list, depending on its dimensions.
        return self.__veri

    def duzlestir(self):
    #Converts the gergen object's multi-dimensional structure into a 1D structure, effectively 'flattening' the object.
        if self.__unnested_veri is None:
            self.__unnested_veri = unnest_list(self.__veri)
        
        return self.__unnested_veri

    def boyutlandir(self, yeni_boyut):
    #Reshapes the gergen object to a new shape 'yeni_boyut', which is specified as a tuple.
        if not isinstance(yeni_boyut, tuple):
            raise ValueError('yeni_boyut should be a tuple')
        
        current_uzunluk = self.uzunluk()
        yeni_uzunluk = 1

        for dim in yeni_boyut:
            yeni_uzunluk *= dim

        if yeni_uzunluk != current_uzunluk:
            raise ValueError('The new shape should have the same number of elements as the original shape')

        unnested_list = self.duzlestir()

        return gergen(nest_list(unnested_list, yeni_boyut))

    def ic_carpim(self, other):
    #Calculates the inner (dot) product of this gergen object with another.
        """
        ic_carpim is defined for 1D and 2D gergens. dis_carpim is only for 1D gergens. topla and ortalama applies for n-dimensional gergens
        see: https://odtuclass2023s.metu.edu.tr/mod/forum/discuss.php?d=757

        Executes the inner prod- uct operation between two gergen objects. This method adheres to the mathematical definition of the inner
        product, which requires both operands to be of the same di- mension.
            For 1-D Tensors: both tensors must have the same dimensionality.
            Matrix Product Representation: In cases where the gergen objects are treated as column vectors, the inner product can be
            expressed through the matrix product x . y = xT y,
            where xT denotes the transpose of vector x.
            
        Error Handling:
            If either operand is not a gergen object, an error is raised to maintain type consistency, ensuring that the operation is
            performed between two gergen instances.
            For 1-D tensors, if the lengths of the vectors do not match, an error is thrown, emphasizing the requirement for equal
            dimensionality in the inner product computation.
            In the case of 2-D tensors, if the number of columns in the first matrix does not equal the number of rows in the second, an
            error is raised, reflecting the necessity for compatible dimensions in matrix multiplication.
        """

        if not isinstance(other, gergen):
            raise TypeError('Operands should be of type gergen')
        
        if len(self.__boyut) != len(other.__boyut):
            raise ValueError('Operands should have the same shape')
        
        if len(self.__boyut) == 1:
            if self.uzunluk() != other.uzunluk():
                raise ValueError('Operands should have the same shape')

            return sum([self.__veri[i] * other.__veri[i] for i in range(len(self.__veri))])
        
        if len(self.__boyut) == 2:
            if self.__boyut[1] != other.__boyut[0]:
                raise ValueError('Operands should have compatible dimensions')
            
            n = self.__boyut[0]
            m = self.__boyut[1]
            p = other.__boyut[1]

            result = []

            for i in range(n):
                for j in range(p):
                    result.append(sum([self.__veri[i][k] * other.__veri[k][j] for k in range(m)]))

            return gergen(nest_list(result, (n, p)))
        
        raise ValueError('ic_carpim is only defined for 1D and 2D gergens')


    def dis_carpim(self, other):
    #Calculates the outer product of this gergen object with another.
        """
        ic_carpim is defined for 1D and 2D gergens. dis_carpim is only for 1D gergens. topla and ortalama applies for n-dimensional gergens
        see: https://odtuclass2023s.metu.edu.tr/mod/forum/discuss.php?d=757
        """
            
        if not isinstance(other, gergen):
            raise TypeError('Operands should be of type gergen')

        if len(self.__boyut) != 1 or len(other.__boyut) != 1:
            raise ValueError('Operands should be 1D gergens')
        
        if self.uzunluk() != other.uzunluk():
            raise ValueError('Operands should have the same shape')
        
        result = []

        for i in range(self.uzunluk()):
            for j in range(other.uzunluk()):
                result.append(self.__veri[i] * other.__veri[j])

        return gergen(nest_list(result, (self.uzunluk(), other.uzunluk())))


    def topla(self, eksen=None):
    #Sums up the elements of the gergen object, optionally along a specified axis 'eksen'.
        """
        Adds up values in gergen. If eksen is None, all elements are added. If eksen is not None, you can see the examples below:
            Column-wise Addition (eksen=0): Elements over the vertical axis are added and returned as a gergen with the same size as the
            number of columns.
            Row-wise Addition (eksen=1): Elements over the horizontal axis are added and returned as a gergen with the same size as the
            number of rows.
        Error Handling:
            If the specified eksen is not an integer or None, a TypeError is raised to indicate that eksen must be an integer or None.
            When an eksen is provided, the function verifies that it is within the valid range of the data’s dimensions. If eksen exceeds
            the dimensions, a ValueError is raised indicating that the specified eksen is out of bounds.
        """

        """
        ic_carpim is defined for 1D and 2D gergens. dis_carpim is only for 1D gergens. topla and ortalama applies for n-dimensional gergens.
        see: https://odtuclass2023s.metu.edu.tr/mod/forum/discuss.php?d=757
        """

        """
        the eksen param is the same param as the axis param in numpy's sum function.
        """

        unnested_veri = self.duzlestir()

        if eksen is not None:
            if not isinstance(eksen, int):
                raise TypeError('eksen should be an integer or None')
            
            if eksen < 0 or eksen >= len(self.__boyut):
                raise ValueError('eksen is out of bounds')

            boyut_list = list(self.__boyut)
            boyut_list.pop(eksen)    

            resulting_gergen_veri = []
            resulting_gergen_boyut = tuple(boyut_list)

            """
            about the algorithm:
                what the algorithm does is:
                    - flatten the list.
                    - we will sum N elements in each iteration, where N is self.__boyut[eksen].
                    - the N elements we will sum is going to be every Mth element, where M is the product of all elements in the list
                      self.__boyut[eksen+1:]. If eksen is the last element, then M is 1.
            """

            N = self.__boyut[eksen]
            M = 1

            for el in self.__boyut[eksen+1:]:
                M *= el

            summed_elem_counter = 0
            total_elem_counter = 0
            current_elem_sum = 0
            offset = 0
            offset_check_coefficient = 1

            while total_elem_counter <= self.uzunluk() / N:
                if total_elem_counter >= M * offset_check_coefficient:
                    offset += M * (N - 1)
                    offset_check_coefficient += 1

                for i in range(total_elem_counter + offset, len(unnested_veri), M):
                    current_elem_sum += unnested_veri[i]

                    if summed_elem_counter == N - 1:
                        resulting_gergen_veri.append(current_elem_sum)
                        current_elem_sum = 0
                        summed_elem_counter = 0
                        break

                    summed_elem_counter += 1
                
                total_elem_counter += 1

            return gergen(nest_list(resulting_gergen_veri, resulting_gergen_boyut))

        return sum(unnested_veri)

    def ortalama(self, eksen=None):
    #Calculates the average of the elements of the gergen object, optionally along a specified axis 'eksen'.
        """
        Computes average (mean) of ele- ments in a tensor, with the flexibility to compute this average across different axes of the tensor
        based on the eksen parameter (similar to topla).
        When no eksen parameter is specified, the method computes the overall average of all elements within the tensor, treating it as a
        flattened array. This is akin to calculating the mean value of a set of numbers.
        Error Handling:
            If the specified eksen is not an integer or None, a TypeError is raised to indicate that eksen must be an integer or None.
            When an eksen is provided, the function verifies that it is within the valid range of the data’s dimensions. If eksen exceeds
            the dimensions, a ValueError is raised indicating that the specified eksen is out of bounds.
        """

        """
        ic_carpim is defined for 1D and 2D gergens. dis_carpim is only for 1D gergens. topla and ortalama applies for n-dimensional gergens.
        see: https://odtuclass2023s.metu.edu.tr/mod/forum/discuss.php?d=757
        """

        divisor = self.uzunluk() if eksen is None else self.__boyut[eksen]

        return self.topla(eksen) / divisor

    def map(self, func):
    #Applies a function to each element of the gergen object.
        """
        Applies the specified function to each element in the gergen object, returning a new gergen object with the transformed values.
        The function should be a lambda function or a user-defined function that can be applied to each element of the gergen object.
        """
        return gergen(map_nested_list(self.__veri, func)) 
    

######################################################################
#-------------------- FUNDAMENTAL FUNCTIONALITIES -------------------#
######################################################################
def cekirdek(sayi: int) -> None:
    """
    Sets the seed for random number generation to ensure reproducibility of results. Before generating random numbers (for instance,
    when initializing tensors with random values), you can call this function to set the seed.
    """
    random.seed(sayi)


def rastgele_dogal(boyut: tuple, aralik: tuple = (0, 100), dagilim='uniform') -> gergen:
    """
    Generates a gergen of specified dimensions with random integer values. The boyut parameter is a tuple specifying the dimensions of
    the gergen to be generated. The aralik parameter is an optional tuple (min, max) specifying the range of random values, with a 
    default range of (0, 100). The dagilim parameter specifies the distribution of random values, with ‘uniform’ as the default
    distribution. Possible values for dagilim include ‘uniform’ for a uniform distribution. You should raise ValueError if dagilim
    parameter is given differently.
    """

    if dagilim != 'uniform':
        raise ValueError('dagilim parameter should be uniform')
    
    aralik_list: list = list(aralik)

    # if boyut is 0, then we should return a scalar.
    if len(boyut) == 0:
        random_scalar = random.randint(*aralik_list)

        return gergen(random_scalar)

    return gergen(create_nested_list(boyut, aralik_list, True))


def rastgele_gercek(boyut: tuple, aralik: tuple = (0.0,1.0), dagilim = None) -> gergen:
    """
    Generates a gergen of specified dimensions with random floating-point values. The boyut parameter is a tuple specifying the dimensions
    of the gergen to be generated. The aralik parameter is an optional tuple (min, max) specifying the range of random values, with a
    default range of (0.0,1.0). The dagilim parameter specifies the distribution of random values, with ‘uniform’ as the default
    distribution. Possible values for dagilim include ‘uniform’ for a uniform distribution. You should raise ValueError if dagilim
    parameter is given differently.
    """

    if dagilim != 'uniform' and dagilim is not None:
        raise ValueError('dagilim parameter should be uniform')
    
    aralik_list: list = list(aralik)

    # if boyut is 0, then we should return a scalar.
    if len(boyut) == 0:
        random_scalar = random.uniform(*aralik_list)

        return gergen(random_scalar)
    
    return gergen(create_nested_list(boyut, aralik_list, False))


######################################################################
#--------------------------- TEST EXAMPLES --------------------------#
######################################################################


def example_1():
    """
    Using rastgele gercek, generate two gergen objects A and B with shapes (64, 64) and calculate:

    A^T B

    Then, calculate the same function with NumPy and report the time and difference.
    """
    #Example 1
    boyut = (64,64)
    g1 = rastgele_gercek(boyut)
    g2 = rastgele_gercek(boyut)

    np_arr1 = np.array(g1.listeye())
    np_arr2 = np.array(g2.listeye())

    start = time.time()
    
    calculated = g1.ic_carpim(g2)
    
    end = time.time()

    start_np = time.time()
    
    actual = np.dot(np_arr1, np_arr2)

    end_np = time.time()

    test_results = TestResult(
        durations=(end-start, end_np-start_np),
        test_name="example_1",
        results=(calculated.listeye(), actual),
    )

    print(test_results)

    return test_results


def example_2():
    """
    Using rastgele gercek, generate three gergens A, B and C with shapes (4,16,16,16) and report the time and result with their NumPy
    equivalent:
    (A x B + C x A + B x C).ortalama()
    """
    #Example 2
    boyut = (4, 16, 16, 16)
    g1 = rastgele_gercek(boyut)
    g2 = rastgele_gercek(boyut)
    g3 = rastgele_gercek(boyut)

    np_arr1 = np.array(g1.listeye())
    np_arr2 = np.array(g2.listeye())
    np_arr3 = np.array(g3.listeye())

    start = time.time()
    
    calculated = (g1 * g2 + g3 * g1 + g2 * g3).ortalama()
    
    end = time.time()

    start_np = time.time()
    
    actual = np.average(np_arr1 * np_arr2 + np_arr3 * np_arr1 + np_arr2 * np_arr3)

    end_np = time.time()

    test_results = TestResult(
        durations=(end-start, end_np-start_np),
        test_name="example_2",
        results=(calculated, actual),
    )

    print(test_results)

    return test_results
    


def example_3():
    """
    Using rastgele gercek, generate two gergen’s A and B with shapes (3,64,64) and report
    the time and result with their NumPy equivalent:
    ln ((sin(A) + cos(B))**2 ) / 8
    """
    #Example 3

    boyut = (3, 64, 64)

    g1 = rastgele_gercek(boyut)
    g2 = rastgele_gercek(boyut)

    np_arr1 = np.array(g1.listeye())
    np_arr2 = np.array(g2.listeye())

    start = time.time()

    calculated = ((g1.sin() + g2.cos()).us(2)).ln() / 8

    end = time.time()

    start_np = time.time()

    actual = np.log((np.sin(np_arr1) + np.cos(np_arr2))**2) / 8

    end_np = time.time()

    test_results = TestResult(
        durations=(end-start, end_np-start_np),
        results=(calculated.listeye(), actual),
        test_name="example_3"
    )

    print(test_results)

    return test_results


"""
COMPARISON RESULTS WITH NUMPY

example_1:
    Are the results the same?: True
    Time taken for gergen: 0.03065013885498047
    Time taken for numpy: 0.00010061264038085938

example_2:
    Are the results the same?: True
    Time taken for gergen: 0.034868717193603516
    Time taken for numpy: 0.00017833709716796875

example_3:
    Are the results the same?: True
    Time taken for gergen: 0.02022266387939453
    Time taken for numpy: 0.0003833770751953125
"""










