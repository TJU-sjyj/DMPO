from loss_schedule.gamma import MileStone, Linear, Exponential, Cosine
from loss_schedule.gradnorm import GradNorm

'''
alpha_dict = {
    "type":
    "args": {}
}
gamma_dict = {
    "type":
    "args": {}
}
'''

class Lambda():
    def __init__(self, alpha_dict=None, beta_list=None, gamma_dict=None, gradnorm_dict=None):
        # Gamma
        if gamma_dict is None:
            self.gamma = Linear(use_gamma=False)
        else:
            Gamma = globals()[gamma_dict['type']]
            self.gamma = Gamma(**gamma_dict['args'])
        # beta
        if beta_list is None:
            self.beta = [1, 1, 1, 1]
        else:
            assert len(beta_list) == 4
            self.beta = beta_list
        
        if gradnorm_dict is None:
            self.gradnorm = GradNorm(use_gn=False)
        else:
            GN = globals()[gradnorm_dict['type']]
            self.gradnorm = GN(**gradnorm_dict['args'])
        
        self.lad = [1.0, 1.0, 1.0, 1.0]
        self.update()
    
    def update_gamma(self, epoch):
        self.gamma.update(epoch)
        self.update()
    
    def update_alpha(self, outs, targets):
        losses = self.alpha.update(outs, targets)
        self.update()
        return losses

    def update_gn(self, loss):
        self.gradnorm.update(loss)
        self.update()
    
    def update(self):
        for idx in range(4):
            self.lad[idx] = self.beta[idx] * self.gamma.get(idx) * self.gradnorm.get(idx)

    def get(self, idx):
        return self.lad[idx]