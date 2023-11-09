"""by lyuwenyu
"""

import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import numpy as np 

from src.core import YAMLConfig

import torch
import torch.nn as nn 


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model
            
        def forward(self, images):
            outputs = self.model(images)
            return outputs
    

    model = Model()
    model.eval()


    # import onnxruntime as ort 
    from PIL import Image, ImageDraw
    from torchvision.transforms import ToTensor

    # # print(onnx.helper.printable_graph(mm.graph))

    image = Image.open('./000000001532.jpg').convert('RGB')
    im = image.resize((640, 640))
    im_data = ToTensor()(im)[None]
    print(im_data.shape)

    pred = model(im_data)

    pred["pred_logits"] = torch.sigmoid(pred["pred_logits"])

    

    print(pred["pred_logits"].shape,pred["pred_boxes"].shape)
    argmax = torch.argmax(pred["pred_logits"],2).reshape(-1)
    print(argmax.shape)

    pred_logits = pred["pred_logits"]
    pred_boxes = pred["pred_boxes"]
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
    

    # sess = ort.InferenceSession(args.file_name)
    # output = sess.run(
    #     # output_names=['labels', 'boxes', 'scores'],
    #     output_names=None,
    #     input_feed={'images': im_data.data.numpy(), "orig_target_sizes": size.data.numpy()}
    # )

    # # print(type(output))
    # # print([out.shape for out in output])

    # labels, boxes, scores = output

    # draw = ImageDraw.Draw(im)
    # thrh = 0.6

    # for i in range(im_data.shape[0]):

    #     scr = scores[i]
    #     lab = labels[i][scr > thrh]
    #     box = boxes[i][scr > thrh]

    #     print(i, sum(scr > thrh))

    #     for b in box:
    #         draw.rectangle(list(b), outline='red',)
    #         draw.text((b[0], b[1]), text=str(lab[i]), fill='blue', )

    # im.save('test.jpg')



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )

    args = parser.parse_args()

    main(args)
