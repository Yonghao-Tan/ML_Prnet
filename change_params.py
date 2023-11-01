import torch

net = torch.load('./checkpoints/exp2/models/model.best.t7.blk' ,map_location=torch.device('cuda'))
for key, value in list(net.items()):
    print(key, value.size(), sep=" ")

net['attention.model.encoder.layers.0.norm1.a_2'] = net.pop('attention.model.encoder.layers.0.sublayer.0.norm.a_2')
net['attention.model.encoder.layers.0.norm1.b_2'] = net.pop('attention.model.encoder.layers.0.sublayer.0.norm.b_2')
net['attention.model.encoder.layers.0.norm2.a_2'] = net.pop('attention.model.encoder.layers.0.sublayer.1.norm.a_2')
net['attention.model.encoder.layers.0.norm2.b_2'] = net.pop('attention.model.encoder.layers.0.sublayer.1.norm.b_2')

net['attention.model.decoder.layers.0.norm1.a_2'] = net.pop('attention.model.decoder.layers.0.sublayer.0.norm.a_2')
net['attention.model.decoder.layers.0.norm1.b_2'] = net.pop('attention.model.decoder.layers.0.sublayer.0.norm.b_2')
net['attention.model.decoder.layers.0.norm2.a_2'] = net.pop('attention.model.decoder.layers.0.sublayer.1.norm.a_2')
net['attention.model.decoder.layers.0.norm2.b_2'] = net.pop('attention.model.decoder.layers.0.sublayer.1.norm.b_2')
net['attention.model.decoder.layers.0.norm3.a_2'] = net.pop('attention.model.decoder.layers.0.sublayer.2.norm.a_2')
net['attention.model.decoder.layers.0.norm3.b_2'] = net.pop('attention.model.decoder.layers.0.sublayer.2.norm.b_2')

# net.pop('attention.model.encoder.layers.0.self_attn.linears.3.weight')
# net.pop('attention.model.encoder.layers.0.self_attn.linears.3.bias')
# net.pop('attention.model.decoder.layers.0.self_attn.linears.3.weight')
# net.pop('attention.model.decoder.layers.0.self_attn.linears.3.bias')
# net.pop('attention.model.decoder.layers.0.src_attn.linears.3.weight')
# net.pop('attention.model.decoder.layers.0.src_attn.linears.3.bias')
torch.save(net, './checkpoints/exp2/models/model.best.t7')
# # #验证修改是否成功
net = torch.load('./checkpoints/exp2/models/model.best.t7' ,map_location=torch.device('cuda'))
for key, value in list(net.items()):
    print(key, value.size(), sep=" ")
    