
import torch
import onnx
import os
import sys
# from torch.onnx import register_custom_op_symbolic
# from torch.onnx.symbolic_helper import parse_args
# Register onnx op
# @parse_args('v', 'v', 'v')
# def symbolic(g, input, slopes, intercepts):
#     output = g.op('access::Hardpwl', input, slopes, intercepts, shape_infer_pattern_s="SameAs")
#     output.setType(input.type())
#     return output
# register_custom_op_symbolic("ac::Hardpwl", symbolic, 9)


def pth_to_onnx(input, model, onnx_path, src, tgt, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    input_names = ["src", "tgt"]
    output_names = ["output"]
    model.eval()
    model.to(src.device)
    # print(input.shape)
    import logging
    logging.basicConfig(level=logging.DEBUG)
    with open('onnx_export_log.txt', 'w') as f:
        # 将标准输出和错误输出重定向到文件
        sys.stdout = f
        sys.stderr = f
        
        # 尝试导出模型，并捕获可能发生的异常
        try:
            torch.onnx.export(model, (src, tgt), onnx_path, verbose=True, input_names=input_names, output_names=output_names, opset_version=19)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        
        # 恢复原始的标准输出和错误输出
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    print("Exporting .pth model to onnx model has been successful!")
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# onnx_path = './segformer_b0_full_precision.onnx'
# onnx_path_simp = './segformer_b0_full_precision_simp.onnx'
# backbone = mit_b0()
# head = SegformerHead(in_channels=[32, 64, 160, 256],
#     in_index=[0, 1, 2, 3],
#     feature_strides=[4, 8, 16, 32],
#     channels=128,
#     dropout_ratio=0.1,
#     num_classes=19,
#     align_corners=False,
#     decoder_params=dict(embed_dim=256),
#     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
# model = nn.Sequential(collections.OrderedDict([
#     ('backbone', backbone),
#     ('decode_head', head),
#     ]))
# checkpoint = torch.load('/nfs/usrhome/pdongaa/Segformer_Quan/SegFormer/pretrained/segformer.b0.1024x1024.city.160k.conv.pth', map_location=torch.device('cpu'))
# if "state_dict" in checkpoint:
#     model.load_state_dict(checkpoint["state_dict"])
# else:
#     model.load_state_dict(checkpoint["model_state"])
# input = tuple([torch.randn(1, 3, 1024, 1024, requires_grad=False)])
# pth_to_onnx(input, model, onnx_path)

# from onnxsim import simplify
# onnx_model = onnx.load(onnx_path)
# model_simp, check = simplify(onnx_model)
# assert check, "Simplified ONNX model could not be validated"
# onnx.save(model_simp, onnx_path_simp)
# print("Simplified onnx model saved at {}".format(onnx_path_simp))