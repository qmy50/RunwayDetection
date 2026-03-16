import torch

pti = [1,2,3,4,-1]
a = [int(pt // 2) if pt != -1 else 255 for pt in pti]
print(a)