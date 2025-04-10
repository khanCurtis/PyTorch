import torch

x = torch.ones(5) #input tensor
y = torch.zeros(3) #expected
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}\n")

#computing gradients
loss.backward()
print(w.grad)
print(b.grad)

#disabling gradient tracking
print(f"Gradient Tracking: {z.requires_grad}") #prints true because it requires grad

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(f"Gradient Tracking: {z.requires_grad}") #prints false because grad has been disabled

#you can also disable like this
z_det = z.detach()
print(f"Gradient Tracking: {z_det.requires_grad}") #prints false because grad is detatched

#tensor gradients and jacobian products
inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"Call after zeroing gradients\n{inp.grad}")