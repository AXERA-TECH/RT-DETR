import onnx
from onnx import helper

model_path = "./rtdetr_pytorch/weights/rtdetr_r18vd_5x_coco_objects365_from_paddle.onnx"
model_path_output = "./rtdetr_pytorch/weights/rtdetr_r18vd_5x_coco_objects365_from_paddle_opt.onnx"
model = onnx.load(model_path)


for node in model.graph.node:
    if node.op_type == "AveragePool":
        print("hello")

        node.attribute.extend([helper.make_attribute("count_include_pad", 1)])

onnx.save(model, model_path_output)
