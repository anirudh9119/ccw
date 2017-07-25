from theano.gof import Op
import theano.tensor as tensor
import theano
from theano import scalar as scal
import numpy as np
from theano import gof
from theano.gradient import DisconnectedType
from theano.tensor.var import  TensorConstant, TensorVariable
from theano.tensor.basic import get_vector_length, NotScalarConstantError, maximum, mul, switch, eq, as_tensor_variable
int_dtypes = list(map(str, scal.int_types))
from theano.compile import shape
#from theano.gof.type import Generic
from theano.gof import ParamsType
from theano.scalar import Scalar

#from theano.scalar import as_scalar


class Reshape(Op):
    """Perform a reshape operation of the input x to the new shape shp.
    The number of dimensions to which to reshape to (ndim) must be
    known at graph build time.
    """
    view_map = {0: [0]}  # output 0 is potentially aliased to inputs [0]
    _f16_ok = True

    check_input = False
    __props__ = ("ndim",)
    params_type = ParamsType(ndim=Scalar('int32'))
    # name does not participate because it doesn't affect computations

    def __init__(self, ndim, name=None):
        self.ndim = int(ndim)
        if ndim < 0:
            raise ValueError("The output dimensions after reshape must be 0 or greater")
        assert name is None, 'name attribute for Reshape has been deprecated'

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.ndim)


    def get_params(self, node):
         return int(self.ndim)

    def make_node(self, x, shp):
        x = as_tensor_variable(x)
        shp_orig = shp
        shp = as_tensor_variable(shp, ndim=1)
        if not (shp.dtype in int_dtypes or
                (isinstance(shp, TensorConstant) and shp.data.size == 0)):
            # It raises an error if shp is not of integer type,
            # except when shp is constant and empty
            # (in this case, shp.dtype does not matter anymore).
            raise TypeError("Shape must be integers", shp, shp.dtype)
        assert shp.ndim == 1
        if isinstance(shp, TensorConstant):
            bcast = [s == 1 for s in shp.data]
            return gof.Apply(self, [x, shp], [tensor(x.type.dtype, bcast)])
        else:
            bcasts = [False] * self.ndim
            shp_list = shp_orig
            if hasattr(shp_orig, "ndim") and shp_orig.ndim == 0:
                shp_list = [shp_orig]
            for index in xrange(self.ndim):
                y = shp_list[index]
                y = as_tensor_variable(y)
                # Try to see if we can infer that y has a constant value of 1.
                # If so, that dimension should be broadcastable.
                try:
                    bcasts[index] = (
                        hasattr(y, 'get_scalar_constant_value') and
                        y.get_scalar_constant_value() == 1)
                except NotScalarConstantError:
                    pass
            return gof.Apply(self, [x, shp], [tensor(x.type.dtype, bcasts)])

    def perform(self, node, inp, out_, params):
        x, shp = inp
        out, = out_
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform has incorrect'
                             ' length %i'
                             ', should be %i' % (len(shp), self.ndim), shp)
        try:
            out[0] = np.reshape(x, shp)
        except Exception:
            raise ValueError('Cannot reshape input of shape %s to shape %s' %
                             (x.shape, shp))

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, shp = inp
        g_out, = grads
        return [reshape(g_out, shape(x), ndim=x.ndim),
                DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], **dict(return_list=True))

    def infer_shape(self, node, ishapes):
        # inputs[1] can contain at most one value of '-1', meaning the actual
        # shape of the output will be automatically computed by reshape, so
        # that the total number of elements stays the same.
        # TODO: Maybe put that formula here?
        # It's not trivial, because we would have to check if the product of
        # all the non-minus-one shapes is a divisor of the product of the
        # original shapes.

        # The following expression leads to cycles in feature_shape,
        # because it tries to replace the Shape_i node by the switch
        # statement, which depends on Shape_i.
        # return [tuple([switch(eq(node.inputs[1][i], -1),
        #                      theano.tensor.opt.Shape_i(i)(node.outputs[0]),
        #                      node.inputs[1][i])
        #                    for i in xrange(self.ndim)]
        #    )]

        # Here, we only simplify if the shape (node.inputs[1]) is a constant,
        # ideally it would suffice to check that it is always non-negative.

        # If current variable is a scalar and its dimensionality should
        # change to self.ndim, then use size 1 for all new dimensions.
        if len(ishapes[0]) == 0:
            return [(1,) * self.ndim]

        requ = node.inputs[1]
        input_size = mul(*ishapes[0])
        if isinstance(requ, theano.tensor.TensorConstant):
            requ = list(requ.data)
            requ_part = [ele for ele in requ if ele != -1]
            crit = len(requ) - len(requ_part)
            if crit == 1 and len(requ_part) > 0:
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                requ_size = mul(*requ_part)
                missing = input_size // (1 if requ_size == 0 else requ_size)
                for i, ele in enumerate(requ):
                    if ele == -1:
                        requ[i] = missing
            elif crit == 1:  # we reshape to -1
                requ = [input_size] if ishapes[0] else [1]
            elif crit > 1:
                raise ValueError('shape argument to Reshape.perform'
                                 ' must have at most one entry equal to -1')
            return [requ]
        else:
            requ = [requ[i] for i in xrange(self.ndim)]
            # since new_dims can have negative value (-1), the
            # multiplication of all values should be negated
            # to give a positive value.
            # To avoid optimization complexity, we avoid checking
            # for the case when there are two or more '-1' values.
            if self.ndim:
                requ_size = -mul(*requ)
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                rest_size = input_size // maximum(requ_size, 1)
            return [tuple([switch(eq(requ[i], -1),
                                  rest_size,
                                  requ[i])
                           for i in xrange(self.ndim)])]

    def c_code_cache_version(self):
        return (7,)

    def c_code(self, node, name, inputs, outputs, sub):
        if isinstance(node.inputs[0], TensorVariable):
            x, shp = inputs
            z, = outputs
            new_ndim = self.ndim
            sdtype = node.inputs[1].type.dtype_specs()[1]
            fail = sub['fail']
            return """
            assert (PyArray_NDIM(%(shp)s) == 1);
            npy_intp new_dims[%(params)s->ndim];
            PyArray_Dims newshape;
            newshape.ptr = new_dims;
            newshape.len = %(params)s->ndim;
            for (int ii = 0; ii < %(params)s->ndim; ++ii)
            {
                // -- We do not want an explicit cast here. the shp can be any
                // -- int* dtype. The compiler will explicitly upcast it, but
                // -- will err if this will downcast. This could happen if the
                // -- user pass an int64 dtype, but npy_intp endup being int32.
                new_dims[ii] = ((%(sdtype)s*)(
                        PyArray_BYTES(%(shp)s) +
                        ii * PyArray_STRIDES(%(shp)s)[0]))[0];
            }
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject *) PyArray_Newshape(%(x)s, &newshape, NPY_CORDER);
            if (!%(z)s)
            {
                //The error message should have been set by PyArray_Newshape
                %(fail)s;
            }
            """ % locals()
        else:
            return Op.c_code(self, node, name, inputs, outputs, sub)


def reshape(x, newshape, ndim=None):
    if ndim is None:
        newshape = as_tensor_variable(newshape)
        if newshape.ndim != 1:
            raise TypeError(
                "New shape in reshape must be a vector or a list/tuple of"
                " scalar. Got %s after conversion to a vector." % newshape)
        try:
            ndim = get_vector_length(newshape)
        except ValueError:
            raise ValueError(
                "The length of the provided shape (%s) cannot "
                "be automatically determined, so Theano is not able "
                "to know what the number of dimensions of the reshaped "
                "variable will be. You can provide the 'ndim' keyword "
                "argument to 'reshape' to avoid this problem." % newshape)
    op = Reshape(ndim)
    rval = op(x, newshape)
    return rval
