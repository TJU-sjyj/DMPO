import torch
from torch.optim import Adam
import torch.distributed as dist

class GradNorm():
    def __init__(self, lr=0.0, para=None, alpha=1.0, opt=Adam, use_gn=True, use_dist=False):
        # alpha
        self.use_gn = use_gn
        self.use_dist = use_dist
        if not use_gn:
            self.weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.lr = lr
            self.opt_type = opt
            self.weights = torch.ones(4)
            self.weights = torch.nn.Parameter(self.weights)
            self.opt = self.opt_type([self.weights], lr=self.lr)
            self.T = self.weights.sum().detach()
            self.para = para
            self.is_init = True
            self.alpha = alpha

    def update(self, loss):
        # loss 先将loss1-4 stack起来
        if not self.use_gn:
            return
        loss = torch.stack(loss)
        # print(f"\n\n\n----------loss: {loss}")
        if self.is_init:
            self.l0 = loss.detach()
            if self.use_dist:
                torch.distributed.all_reduce(self.l0)
                self.l0 = self.l0 / (dist.get_world_size())
            if (self.l0 == 0).any():
                average_value = self.l0.mean()
                new_tensor = torch.full_like(self.l0, average_value.item(), device=self.l0.device)
                self.l0 = new_tensor
            self.is_init = False
        gw = []
        for i in range(4):
            dl = torch.autograd.grad(loss[i], self.para, retain_graph=True, create_graph=True)[0]
            # if i == 3:
                # print(f"Before----------rank{dist.get_rank()}--dl_{i}: {dl}")
            if self.use_dist:
                torch.distributed.all_reduce(dl)
                # print(f"After----------rank{dist.get_rank()}--dl_{i}: {dl}")
                # exit()
            gw.append(torch.norm(dl))
        gw = torch.stack(gw)
        # print(f"----------gw: {gw}")
        if self.use_dist:
            torch.distributed.all_reduce(loss)
            loss = loss / (dist.get_world_size())
        
        loss_ratio = loss.detach() / self.l0
        # print(f"----------loss_ratio: {loss_ratio}")
        rt = loss_ratio / loss_ratio.mean()
        # print(f"----------rt: {rt}")
        gw_avg = gw.mean().detach()
        constant = (gw_avg * rt ** self.alpha).detach()
        
        # print(f"----------constant: {constant}")
        gradnorm_loss = torch.abs(gw - constant).sum()
        # print(f"----------gradnorm_loss: {gradnorm_loss}")
        self.opt.zero_grad()
        gradnorm_loss.backward()
        # print(f"Before----------self.weights: {self.weights}")
        # print(f"***Before----------rank{dist.get_rank()}--self.weights: {self.weights}")
        self.opt.step()
        # print(f"After----------rank{dist.get_rank()}--self.weights: {self.weights}")
        # exit()
        # print(f"After----------self.weights: {self.weights}")
        
        self.weights = (self.weights / self.weights.sum() * self.T).detach()
        self.weights = torch.nn.Parameter(self.weights)
        self.opt = torch.optim.Adam([self.weights], lr=self.lr)

    def get(self, idx):
        return self.weights[idx]