import torch

class Gaussian():
    def __init__(self, device, mean, std):
        self.device = device
        self.mean = mean
        self.std = std

    def apply(self, X, saliency):
        noise = torch.empty(*X.shape, device=self.device)
        noise.normal_(self.mean, self.std)
        # low saliency score == important
        X_perturbed = X + (saliency * noise)
        # X_perturbed.to(self.device)
        return X_perturbed
