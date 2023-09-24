
from onnxmltools.utils import load_model


onnx_model = load_model("my_ppo_model.onnx")

print(onnx_model)