#!/usr/bin/env python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Load model
graph = gs.import_onnx(onnx.load("pfld-sim.onnx"))


# Find original input tensor
orig_in = graph.inputs[0]

# new input tensor for uint8 image data
img_u8 = gs.Variable(name="img_u8", dtype=np.uint8, shape=orig_in.shape)
graph.inputs = [img_u8]

# Temp variable between cast and rescale
cast_out = gs.Variable(name="cast_out", dtype=np.float32, shape=img_u8.shape)

graph.nodes += [
  # Cast u8 to fp
  gs.Node(op="Cast", inputs=[img_u8], outputs=[cast_out], attrs={'to': onnx.TensorProto.FLOAT}),

  # Rescale from u8 range to 0...1 (divide by 255.0)
  gs.Node(op="Div", inputs=[cast_out, gs.Constant(name="constant_255", values=np.full((1), 255.0, dtype=np.float32))], outputs=[orig_in]),
]

# Save and exit
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "pfld-uint8-in.onnx")

