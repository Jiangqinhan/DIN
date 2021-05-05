from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"

_WEIGHTS_VARIABLE_NAME = "kernel"


class _Linear_:
    def __init__(self, args, output_size, build_bias, bias_initializer=None, kernel_initializer=None):
        '''

        :param args: args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size: int, second dimension of weight variable.在gru中可以用来同时计算多个门的参数因此
        :param build_bias:
        :param bias_initializer:
        :param kernel_initializer:
        '''
        self.build_bias = build_bias
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError('args must be specified')
        self.args = args
        if nest.is_sequence(self.args):
            self.is_sequence = True
        else:
            self.is_sequence = False
            self.args = [args]
        total_size = 0
        arg_shapes = [x.get_shape() for x in self.args]
        for arg_shape in arg_shapes:
            if arg_shape.ndims != 2:
                raise ValueError(
                    "linear is expecting 2D arguments: %s" % arg_shape)
            if arg_shape[1] is None:
                raise ValueError("linear expects shape[1] to be provided for shape %s, "

                                 "but saw %s" % (arg_shape, arg_shape[1])
                                 )
            total_size += arg_shape[1]
        if kernel_initializer is None:
            kernel_initializer = init_ops.glorot_normal_initializer(seed=1024)
        dtype = self.args[0].dtype
        scope = vs.get_variable_scope()
        with vs.variable_scope(scope) as out_scope:
            self.weight = vs.get_variable(name=_WEIGHTS_VARIABLE_NAME, shape=[total_size, output_size],
                                          initializer=kernel_initializer, dtype=dtype)
            if self.build_bias:
                if bias_initializer is None:
                    bias_initializer = init_ops.constant_initializer(0.0)
                with vs.variable_scope(out_scope) as inner_scope:
                    self.bias = vs.get_variable(name=_BIAS_VARIABLE_NAME, shape=[output_size, ],
                                                initializer=bias_initializer, dtype=dtype)

    def __call__(self, args):
        '''

        :param args:
        :return:(batch_size,output_size)
        '''
        if not self.is_sequence:
            args = [args]
        if len(args) == 1:
            res = math_ops.matmul(args[0], self.weight)
        else:
            res = math_ops.matmul(array_ops.concat(args), self.weight)
        if self.build_bias:
            res = nn_ops.bias_add(res, self.bias)
        return res


class AGRUCell(RNNCell):
    def __init__(self, num_units, activation=None, kernel_initializer=None, bias_initializer=None, reuse=None):
        '''

        :param num_units: 输出维数
        :param activation:
        :param kernel_initializer:
        :param bias_initializer:
        :param reuse (optional) Python boolean describing whether to reuse variables

       in an existing scope.  If not `True`, and the existing scope already has

       the given variables, an error is raised.

        _gate_linear:产生gate的系数例如 在GRU中计算u_t和r_t
        _candidate_linear 计算输出h'_t  用于计算最终的h_t
        '''
        self.num_units = num_units
        self.activation = activation if activation is not None else math_ops.tanh
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super(AGRUCell, self).__init__(_reuse=reuse)
        self._gate_linear = None
        self._candidate_linear = None

    # 两个必须的函数
    @property
    def state_size(self):
        return self.num_units

    @property
    def output_size(self):
        return self.num_units

    def call(self, input, state, att_score=None):
        '''

        :param input:
        :param state:
        :param att_score: 注意力机制的输入 [batch_size,1]
        :return: (batch_size,unit_num)
        这个实现很傻逼 build函数的设计果然有它的独到之处
        '''
        if att_score is None:
            raise ValueError("AGRU need att_score")
        if self._gate_linear is None:
            if self.bias_initializer is None:
                self.bias_initializer = init_ops.constant_initializer(1.0, dtype=input.dtype)
            with vs.variable_scope("gate") as current_scope:
                self._gate_linear = _Linear_([input, state], 2 * self.num_units, True, self.bias_initializer,
                                             self.kernel_initializer)
        tmp = self._gate_linear([input, state])
        tmp = math_ops.sigmoid(tmp)
        u, r = array_ops.split(tmp, num_or_size_splits=2, axis=1)
        r_state = state * r
        if self._candidate_linear is None:
            with vs.variable_scope("candidate") as current_scope:
                self._candidate_linear = _Linear_([input, r_state], self.num_units, True, self.bias_initializer,
                                                  self.kernel_initializer)
        c = self.activation(self._candidate_linear([input, r_state]))
        new_h = att_score * c + (1 - att_score) * state
        return new_h, new_h


class AUCRUCell(RNNCell):
    '''
    完全与上面类似
    '''

    def __init__(self,

                 num_units,

                 activation=None,

                 reuse=None,

                 kernel_initializer=None,

                 bias_initializer=None):

        super(AUCRUCell, self).__init__(_reuse=reuse)

        self._num_units = num_units

        self._activation = activation or math_ops.tanh

        self._kernel_initializer = kernel_initializer

        self._bias_initializer = bias_initializer

        self._gate_linear = None

        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""

        if att_score is None:
            raise ("AUGRU needs att_score")
        if self._gate_linear is None:

            bias_ones = self._bias_initializer

            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(
                    1.0, dtype=inputs.dtype)

            with vs.variable_scope("gates"):  # Reset gate and update gate.

                self._gate_linear = _Linear_(

                    [inputs, state],

                    2 * self._num_units,

                    True,

                    bias_initializer=bias_ones,

                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear_(

                    [inputs, r_state],

                    self._num_units,

                    True,

                    bias_initializer=self._bias_initializer,

                    kernel_initializer=self._kernel_initializer)

        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = att_score * u
        new_h = (1 - u) * state + u * c
        return new_h, new_h
