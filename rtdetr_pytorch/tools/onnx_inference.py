import onnxruntime
import glob


class onnx_inferencer:

    def __init__(self, model_path) -> None:
        self.onnx_model_sess = onnxruntime.InferenceSession(model_path)
        self.output_names = []
        self.input_names = []
        print(model_path)
        for i in range(len(self.onnx_model_sess.get_inputs())):
            self.input_names.append(self.onnx_model_sess.get_inputs()[i].name)
            print("    input:", i,
                  self.onnx_model_sess.get_inputs()[i].name,
                  self.onnx_model_sess.get_inputs()[i].shape)

        for i in range(len(self.onnx_model_sess.get_outputs())):
            self.output_names.append(
                self.onnx_model_sess.get_outputs()[i].name)
            print("    output:", i,
                  self.onnx_model_sess.get_outputs()[i].name,
                  self.onnx_model_sess.get_outputs()[i].shape)
        print("")

    def get_input_count(self):
        return len(self.input_names)

    def get_input_shape(self, idx: int):
        return self.onnx_model_sess.get_inputs()[idx].shape

    def get_input_names(self):
        return self.input_names

    def get_output_count(self):
        return len(self.output_names)

    def get_output_shape(self, idx: int):
        return self.onnx_model_sess.get_outputs()[idx].shape

    def get_output_names(self):
        return self.output_names

    def inference(self, tensor):
        return self.onnx_model_sess.run(
            self.output_names, input_feed={self.input_names[0]: tensor})

    def inference_multi_input(self, tensors: list):
        inputs = dict()
        for idx, tensor in enumerate(tensors):
            inputs[self.input_names[idx]] = tensor
        return self.onnx_model_sess.run(input_feed=inputs)


if __name__ == "__main__":
    onnx_model_path = "./weights/tr-detr-gridsample.onnx"
    test_model = onnx_inferencer(onnx_model_path)

        # import onnxruntime as ort 
    from PIL import Image, ImageDraw
    from torchvision.transforms import ToTensor
    import numpy as np
    import torch

    # # print(onnx.helper.printable_graph(mm.graph))

    image = Image.open('./000000001532.jpg').convert('RGB')
    im = image.resize((640, 640))
    im_data = ToTensor()(im)[None]
    print(im_data.shape)

    pred_logits,pred_boxes = test_model.inference(im_data.numpy())

    pred_logits = torch.Tensor(pred_logits)
    pred_boxes = torch.Tensor(pred_boxes)
    print(pred_boxes.shape,pred_logits.shape)

    
    # pred_logits = 1/(1+np.exp(-pred_logits))

    pred_logits = torch.sigmoid(pred_logits)

    

    # print(pred["pred_logits"].shape,pred["pred_boxes"].shape)
    argmax = torch.argmax(pred_logits,2).reshape(-1)
    print(argmax.shape)

    # pred_logits = pred["pred_logits"]
    # pred_boxes = pred["pred_boxes"]
    draw = ImageDraw.Draw(image)

    for i,idx in enumerate(argmax):
        score = pred_logits[0,i,idx]
        if score > 0.6:
            # print(score,idx)
            bbox = pred_boxes[0,i]
            # print(bbox)
            cx,cy,w,h = bbox
            x0 = (cx-0.5*w)*image.width
            y0 = (cy-0.5*h)*image.height
            x1 = (cx+0.5*w)*image.width
            y1 = (cy+0.5*h)*image.height
            draw.rectangle([x0,y0,x1,y1],outline="red")
    image.save("res.jpg")
    

