# count_include_pad 为 0\False 的时候，会产生子图影响推理


import onnx
from onnx import helper

def add_attrubute(model_path):
    print(model_path)
    model = onnx.load(model_path)


    for node in model.graph.node:
        if node.op_type == "AveragePool":
            print("hello")

            node.attribute.extend([helper.make_attribute("count_include_pad", 1)])

    onnx.save(model, model_path)

add_attrubute("weights/rtdetr_r18vd_5x_coco_objects365_from_paddle.onnx")
add_attrubute("weights/rtdetr_r50vd_6x_coco_from_paddle.onnx")
add_attrubute("weights/rtdetr_r101vd_6x_coco_from_paddle.onnx")