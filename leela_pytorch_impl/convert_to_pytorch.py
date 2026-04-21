import sys
import os
import gzip
import torch
import numpy as np

import net_pb2

def decode_layer(layer, name):
    dims = list(layer.dims)
    if not dims and not layer.params:
        return None
        
    encoding = layer.encoding
    params = layer.params
    
    is_linear16 = encoding == net_pb2.Weights.Layer.LINEAR16 or (encoding == 0 and layer.max_val != layer.min_val and len(params) > 0 and len(params) % 2 == 0)
    
    if is_linear16:
        uint16_data = np.frombuffer(params, dtype=np.uint16)
        theta = uint16_data.astype(np.float32) / 65535.0
        data = layer.min_val * (1.0 - theta) + layer.max_val * theta
    elif encoding == net_pb2.Weights.Layer.FLOAT32:
        data = np.frombuffer(params, dtype=np.float32)
    elif encoding == net_pb2.Weights.Layer.FLOAT16:
        data = np.frombuffer(params, dtype=np.float16).astype(np.float32)
    else:
        # Ambiguous encoding (0)
        # Often BN parameters are stored as Float16 (2 bytes per param).
        # We can check if it aligns better with Float16 or Float32.
        # But honestly, everything except explicit float32 is likely Float16 if it didn't trigger Linear16.
        # Let's assume float16 unless len is very clearly float32 only (impossible if divisible by 4, it's also divisible by 2).
        # Wait, if layer is a BN param, it should have the same size as input channels.
        # For simplicity, if we don't know, we guess float16.
        data = np.frombuffer(params, dtype=np.float16).astype(np.float32)
            
    tensor = torch.tensor(data.copy(), dtype=torch.float32)
    if dims:
        tensor = tensor.view(*dims)
    return tensor

def extract_weights(message, prefix=''):
    state_dict = {}
    for field_desc, value in message.ListFields():
        name = field_desc.name
        full_name = f"{prefix}.{name}" if prefix else name
        
        if field_desc.type == field_desc.TYPE_MESSAGE:
            if field_desc.message_type.name == 'Layer':
                if getattr(field_desc, "label", getattr(field_desc, "is_repeated", None)) == field_desc.LABEL_REPEATED:
                    for i, item in enumerate(value):
                        tensor = decode_layer(item, f"{full_name}.{i}")
                        if tensor is not None:
                            state_dict[f"{full_name}.{i}"] = tensor
                else:
                    tensor = decode_layer(value, full_name)
                    if tensor is not None:
                        state_dict[full_name] = tensor
            else:
                if getattr(field_desc, "label", getattr(field_desc, "is_repeated", None)) == field_desc.LABEL_REPEATED:
                    for i, item in enumerate(value):
                        state_dict.update(extract_weights(item, prefix=f"{full_name}.{i}"))
                else:
                    state_dict.update(extract_weights(value, prefix=full_name))
    return state_dict

def convert(pb_gz_path, out_pt_path):
    with gzip.open(pb_gz_path, 'rb') as f:
        data = f.read()
    
    net = net_pb2.Net()
    net.ParseFromString(data)
    
    state_dict = extract_weights(net.weights)
    
    torch.save(state_dict, out_pt_path)
    print(f"Saved {len(state_dict)} tensors to {out_pt_path}")

if __name__ == '__main__':
    convert('791556.pb.gz', 'lc0_weights_791556.pt')
