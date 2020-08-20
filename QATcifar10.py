from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F


QTensor = namedtuple('QTensor', ['tensor', 'scale','zero_point'])

def visualise(x, axs):
    x = x.view(-1).cpu().numpy()
    axs.hist(x)
    

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
  # Calc Scale and zero point of next 
  qmin = 0.
  qmax = 2.**num_bits - 1.

  scale = (max_val - min_val) / (qmax - qmin)

  initial_zero_point = qmin - min_val / scale
  
  zero_point = 0
  if initial_zero_point < qmin:
      zero_point = qmin
  elif initial_zero_point > qmax:
      zero_point = qmax
  else:
      zero_point = initial_zero_point

  zero_point = int(zero_point)

  return scale, zero_point

def calcScaleZeroPointSym(min_val, max_val, num_bits = 8):
    # calc Scale
    max_val = max(abs(min_val),abs(max_val))
    qmin = 0.
    qmax = 2.**(num_bits - 1) -1.
    
    scale = max_val / qmax
    
    return scale, 0


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
      min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def quantize_tensor_sym(x, num_bits = 8, min_val = None, max_val = None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()
        
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2.**(num_bits-1) - 1.
    
    scale = max_val / qmax
    
    q_x = x / scale
    
    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale = scale, zero_point=0)

def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())

def quantizeLayer(x, layer, stat, scale_x, zp_x, vis = False, axs = None, X = None, y = None, sym=False, num_bits = 8):
    W = layer.weight.data
    B = layer.bias.data
    
    if sym:
        w = quantize_tensor_sym(layer.weight.data, num_bits = num_bits)
        b = quantize_tensor_sym(layer.bias.data, num_bits = num_bits)
    else:
        w = quantize_tensor(layer.weight.data, num_bits = num_bits)
        b = quantize_tensor(layer.weight.data, num_bits = num_bits)
        
    layer.weight.data = w.tensor.float()
    layer.bias.data = b.tensor.float()
    
    if vis:
        axs[X,y].set_xlabel("Visualising weights of layer: ")
        visualise(layer.weight.data, axs[X, y])
        
    scale_w = w.scale
    zp_w = w.zero_point
    scale_b = b.scale
    zp_b = b.zero_point
    
    if sym:
        scale_next, zero_point_next = calcScaleZeroPointSym(min_val = stat['min'], max_val = stat['max'])
    else:
        scale_next, zero_point_next = calcScaleZeroPoint(min_val = stat['min'], max_val = stat['max'])
        
    if sym:
        X = x.float()-zp_x
        layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data)
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data)
    else:
        X = x.float() - zp_x
        layer.weight.data = ((scale_x * scale_w) / scale_next) * (layer.weight.data - zp_w)
        layer.bias.data = (scale_b / scale_next) * (layer.bias.data + zp_b)
        
    if sym:
        x = (layer(X))
    else:
        x = (layer(X)) + zero_point_next
        
    x.round_()
    
    x = F.leaky_relu(x)
    
    layer.weight.data = W
    layer.bias.data = B
    
    return x, scale_next, zero_point_next


# Get Min and max of x tensor, and stores it
def updateStats(x, stats, key):
    max_val, _ = torch.max(x,dim=1)
    min_val, _ = torch.min(x,dim=1)
    
    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
        
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1
        
    weighting = 2.0 / (stats[key]['total']) + 1
    
    if 'ema_min' in stats[key]:
        stats[key]['ema_min'] = weighting * (min_val.mean().item()) + (1 - weighting) * stats[key]['ema_min']
    else:
        stats[key]['ema_min'] = weighting * (min_val.mean().item())
        
    if 'ema_max' in stats[key]:
        stats[key]['ema_max'] = weighting * (max_val.mean().item()) + (1 - weighting) * stats[key]['ema_max']
    else:
        stats[key]['ema_max'] = weighting * (max_val.mean().item())
        
    stats[key]['min_val'] = stats[key]['min'] / stats[key]['total']
    stats[key]['max_val'] = stats[key]['max'] / stats[key]['total']
    
    return stats
        

# Reworked Forward Pass to access activation Stats through updateStats function
def gatherActivationStats(model, x, stats):
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')
    x = F.relu(model.conv1(x))
    x = model.bn1(x)

    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')
    x = F.relu(model.conv2(x))
    x = model.bn2(x)
    x = F.max_pool2d(x,2,2)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')
    x = F.relu(model.conv3(x))
    x = model.bn3(x)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')
    x = F.relu(model.conv4(x))
    x = model.bn4(x)
    x = F.max_pool2d(x,2,2)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')
    x = F.relu(model.conv5(x))
    x = model.bn5(x)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv6')
    x = F.relu(model.conv6(x))
    x = model.bn6(x)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv7')
    x = F.relu(model.conv7(x))
    x = model.bn7(x)
    x = F.max_pool2d(x,2,2)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv8')
    x = F.relu(model.conv8(x))
    x = model.bn8(x)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv9')
    x = F.relu(model.conv9(x))
    x = model.bn9(x)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv10')
    x = F.relu(model.conv10(x))
    x = model.bn10(x)
    x = F.max_pool2d(x,2,2)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv11')
    x = F.relu(model.conv11(x))
    x = model.bn11(x)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv12')
    x = F.relu(model.conv12(x))
    x = model.bn12(x)
    
    stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv13')
    x = F.relu(model.conv13(x))
    x = model.bn13(x)
    x = F.max_pool2d(x,2,2)
    
    x = x.view(x.size(0),-1)
    
    stats = updateStats(x, stats, 'fc1')
    
    x = model.fc1(x)
    
    return stats


# Entry function to get stats of all functions.
def gatherStats(model, test_loader):
    device = 'cuda'
    
    model.eval()
    stats = {}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            stats = gatherActivationStats(model, data, stats)
            
    final_stats = {}
    for key, value in stats.items():
        final_stats[key] = {"max" : value["max"] / value["total"], "min" : value["min"] / value["total"],
                            "ema_min" : value["ema_min"], "ema_max" : value["ema_max"]}
        
    return final_stats

class FakeQuantOp(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, num_bits = 8, min_val = None, max_val = None):
        x = quantize_tensor(x, num_bits = num_bits, min_val = min_val, max_val = max_val)
        x = dequantize_tensor(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_output, num_bits = 8, min_val = None, max_val = None):
        x = grad_output
        x = quantize_tensor(x, num_bits =num_bits, min_val = min_val, max_val = max_val)
        x = dequantize_tensor(x)
        grad_output = x
        return grad_output, None, None, None

x = torch.tensor([1,2,3,4]).float()
print(FakeQuantOp.apply(x))