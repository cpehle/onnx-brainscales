import onnx
from onnx import helper as ox
from onnx import AttributeProto, TensorProto, GraphProto, numpy_helper

import numpy as np

seq_length = 100
batch_size = 10
input_size = 10
hidden_size = 5
output_size = 10

#
X0 = ox.make_tensor_value_info('X0', TensorProto.EVENT, [seq_length, batch_size, input_size])
W0 = ox.make_tensor_value_info('W0', TensorProto.FLOAT, [input_size, hidden_size])

R0 = ox.make_tensor_value_info('R0', TensorProto.FLOAT, [hidden_size, hidden_size])
Y0 = ox.make_tensor_value_info('Y0', TensorProto.EVENT, [seq_length, 1, batch_size, hidden_size])
# 
W1 = ox.make_tensor_value_info('W1', TensorProto.FLOAT, [hidden_size, output_size])
R1 = ox.make_tensor_value_info('R1', TensorProto.FLOAT, [output_size, output_size])
Y1 = ox.make_tensor_value_info('Y1', TensorProto.EVENT, [seq_length, 1, batch_size, output_size])

input_layer = ox.make_node(
    'LIFCell',
    ['X0', 'W0', 'R0'],
    ['Y0'],
)

output_layer = ox.make_node(
    'ADEXCell',
    ['Y0', 'W1', 'R1'],
    ['Y1'],
)

graph_def = ox.make_graph(
    [input_layer, output_layer],
    'test-model',
    [X0,W0,R0,W1,R1],
    [Y1],
    initializer=[
        numpy_helper.from_array(np.random.randn(input_size, hidden_size), name='W0'),
        numpy_helper.from_array(np.random.randn(input_size, hidden_size), name='R0'),
        numpy_helper.from_array(np.random.randn(hidden_size, hidden_size), name='W1'),
        numpy_helper.from_array(np.random.randn(hidden_size, hidden_size), name='R1'),
    ],
)

# node = ox.make_node(
#     'OnHICANN',
#     ['X0', 'W0', 'R0'] + ['W1', 'R1'],
#     ['Y1'],
#     graph = graph_def,
#     hicanns = [12]
# )


model = ox.make_model(graph_def, producer_name='onnx-example')

graph = model.graph

for node in graph.node:
    if node.op_type in [
        'LIFLayer',
        'ADEXLayer',
        'LSNNLayer',
        'LSNNCell',
        'LIFCell',
        'ADEXCell'
        ]:
        print(node.op_type)
        print(node.input)
        print(node.output)

onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
print(onnx.helper.printable_node(output_layer))
onnx.save(model, 'lif_model.onnx')