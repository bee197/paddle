import paddle
import math
from x2paddle.op_mapper.onnx2paddle import onnx_custom_layer as x2paddle_nn

class ONNXModel(paddle.nn.Layer):
    def __init__(self):
        super(ONNXModel, self).__init__()
        self.x2paddle_action_net_weight = self.create_parameter(shape=[3, 512], attr='x2paddle_action_net_weight', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x2paddle_action_net_bias = self.create_parameter(shape=[3], attr='x2paddle_action_net_bias', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x2paddle_value_net_weight = self.create_parameter(shape=[1, 512], attr='x2paddle_value_net_weight', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))
        self.x2paddle_value_net_bias = self.create_parameter(shape=[1], attr='x2paddle_value_net_bias', dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=0.0))

    def forward(self, x2paddle_onnx__Gemm_0):
        x2paddle_action_net_weight = self.x2paddle_action_net_weight
        x2paddle_action_net_bias = self.x2paddle_action_net_bias
        x2paddle_value_net_weight = self.x2paddle_value_net_weight
        x2paddle_value_net_bias = self.x2paddle_value_net_bias
        x2paddle_5_mm = paddle.matmul(x=x2paddle_onnx__Gemm_0, y=x2paddle_action_net_weight, transpose_y=True)
        x2paddle_5 = paddle.add(x=x2paddle_5_mm, y=x2paddle_action_net_bias)
        x2paddle_6_mm = paddle.matmul(x=x2paddle_onnx__Gemm_0, y=x2paddle_value_net_weight, transpose_y=True)
        x2paddle_6 = paddle.add(x=x2paddle_6_mm, y=x2paddle_value_net_bias)
        return x2paddle_5, x2paddle_6

def main(x2paddle_onnx__Gemm_0):
    # There are 1 inputs.
    # x2paddle_onnx__Gemm_0: shape-[1, 512], type-float32.
    paddle.disable_static()
    params = paddle.load(r'/ppo/ppo/train_log/model.pdparams')
    model = ONNXModel()
    model.set_dict(params, use_structured_name=True)
    model.eval()
    out = model(x2paddle_onnx__Gemm_0)
    return out
