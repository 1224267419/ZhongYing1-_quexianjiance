import  torch
pthfile=r'.\123.pth'
net=torch.load(pthfile)
print(net)