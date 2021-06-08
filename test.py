from algo.ppo.core import *
from env import create_train_env
import torch
env = create_train_env(1,1,'complex')
obs_dim = env.observation_space.shape
act_dim = env.action_space.n
net = cnn_net(obs_dim[0] ,act_dim).cuda()

from torch.optim import Adam
optimizer = Adam(net.parameters(), lr=0.01)
optimizer.zero_grad()
x = torch.randn(32,4,84,84).cuda()
criterion = torch.nn.MSELoss()
y_label = torch.randn(32,12).cuda()
y_pred = net(x)
loss = criterion(y_pred,y_label)
loss.backward()
optimizer.step()
for p in net.parameters():
    pass

p_grad_numpy = p.grad.cpu().numpy()   # numpy view of tensor data
p_grad_numpy[:] = np.zeros((12,))

print(p_grad_numpy)

for p in net.parameters():
    pass

print(p_grad_numpy)