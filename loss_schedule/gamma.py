import math

class MileStone():
    def __init__(self, Mile, Eta, Init_Gamma) -> None:
        self.Mile = Mile
        self.Eta = Eta
        self.Gamma = Init_Gamma
    
    def update_stage(self, epoch, mile, gamma, eta):
        for milestone_idx, milestone in enumerate(mile):
            if epoch == milestone:
                gamma = gamma * eta[milestone_idx]
                break
        return gamma
    
    def update(self, epoch):
        for idx in range(4):
            self.Gamma[idx] = self.update_stage(epoch, self.Mile[idx], self.Gamma[idx], self.Eta[idx])
    
    def get(self, idx):
        return self.Gamma[idx]


class Linear():
    def __init__(self, End, Start, Epochs, use_gamma=True) -> None:
        self.use_gamma = use_gamma
        if not use_gamma:
            self.Gamma = [1.0, 1.0, 1.0, 1.0]
        else:
            self.epochs = Epochs
            self.w = []
            for max, min in zip(End, Start):
                self.w.append(float((max - min) / (Epochs - 1)))
            self.b = Start
            self.Gamma = [1.0, 1.0, 1.0, 1.0]
            self.update(0)

    def update(self, epoch):
        if not self.use_gamma:
            return
        for idx in range(4):
            self.Gamma[idx] = self.w[idx] * epoch + self.b[idx]

    def get(self, idx):
        return self.Gamma[idx]


class Exponential():
    def __init__(self, Max, Min, Epochs) -> None:
        self.Epochs = Epochs
        self.Min = Min
        self.Max = Max
        self.Gamma = [1.0, 1.0, 1.0, 1.0]
            
    def update(self, epoch):
        for idx in range(4):
            self.Gamma[idx] = self.Min[idx] * ((float(self.Max[idx]) / self.Min[idx]) ** (epoch / float(self.Epochs)))
    
    def get(self, idx):
        return self.Gamma[idx]

class Cosine():
    def __init__(self, Max, Min, Epochs) -> None:
        self.Epochs = Epochs
        self.Min = Min
        self.Max = Max
            
    def update(self, epoch):
        for idx in range(4):
            cos_inner = (epoch % self.Epochs) / self.Epochs
            factor = (1 - math.cos(cos_inner * math.pi)) / 2
            self.Gamma[idx] = self.Min[idx] + factor * (self.Max[idx] - self.Min[idx])
    
    def get(self, idx):
        return self.Gamma[idx]