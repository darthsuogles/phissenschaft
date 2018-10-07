import torch

"""\
Notice that only when evaluating the whole buffer can it work
If just evaluating the code snippet, you'd have this error:
>> RuntimeError: expected a single top-level function
"""

@torch.jit.script
def RNN(h, x, W_h, U_h, W_y, b_h, b_y):
    y = []
    for t in range(x.size(0)):
        h = torch.tanh(x[t] @ W_h + h @ U_h + b_h)
        y += [torch.tanh(h @ W_y + b_y)]
        if t % 10 == 0:
            print("stats: ", h.mean(), h.var())
    return torch.stack(y), h
