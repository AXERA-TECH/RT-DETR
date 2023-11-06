只有3个文件有修改，分别是 

RT-DETR/rtdetr_pytorch/tools/export_onnx.py
RT-DETR/rtdetr_pytorch/src/zoo/rtdetr/utils.py
RT-DETR/rtdetr_pytorch/src/zoo/rtdetr/rtdetr_decoder.py


其中 
utils.py 里添加了自定义算子 MultiScaleDeformableAttn ，由于 RT-DETR 指定的 torch 版本是 2.0，symbolic 里的内容与 bevformer 定义的时候一点不同，这样写能支持自定义算子在 ort 中的 shape inference

export_onnx.py  中的修改主要是移除了后处理，并且配置 repvgg 导出时不要 fuse conv

rtdetr_decoder.py 中主要做 3 件事，分别是 1.使用 MultiScaleDeformableAttn算子；2. 移除 inverse_sigmoid；3. 修改一个中间变量的数据类型，这个数据类型如果不是 Tensor 会在导出时报错，具体细节忘了。
