import torch

a = torch.ones(2,3,4)

b = torch.ones(4,5)
b = b * 0.5

#print(a)
#print(b)
#print(torch.einsum('abc,cd->abd', a, b))

a = torch.ones(2,3,4)
a[1][0][0] = 4
b = torch.ones(4,2,2)
b = b * 0.5

#print(a)
#print(b)
#print(torch.einsum('abc,zbc->zbc', a, b))


a = torch.ones(2, 2, 16, 16)
#print(a)
for c in range(a.shape[3]):
    a[0][0][c] = a[0][0][c] * (c+1)

print(a)
a = torch.transpose(a, 2, 3)
B_z, Heads_z, H_z, W_z = a.size()
print(H_z//(W_z**.5))
a = a.view(B_z, Heads_z, H_z, int(H_z//(W_z**.5)), int(W_z//(H_z**.5)))
print(a)

#a[2] = a[2] * 3
#a[3] = a[3] * 4
