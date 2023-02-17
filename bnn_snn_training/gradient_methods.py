# Approximate gradient methods introduced in my work. 

import numpy as np
import torch

class GradComputer():
    def __init__(self, net_params):
        self.device = net_params[0].device
        self.shapes = [param.shape for param in net_params]
        
    # Virtual function. Compute approximate gradient given offsets to parameters and sampled losses.
    def compute_grad(self, losses, param_offsets):
        return [None for shape in self.shapes] # Do not use this base function!

########################################################################################
# Compute regression-based gradients for the layers of model given loss samples.
# This is done by fitting a linear regression to the losses given the sampling offsets. 
#
# - losses: The losses associated with each sample. 
#                  It is required that first loss is the loss at the origin.
# - param_offsets: The offsets applied for each sample to the model parameters. 
# - parameters: The parameters (weights, biases, etc) of the original model. 
#                  Only needed for correctly reshaping array. 
# - stddev: Standard deviation used for sampling. Unused in this case.
#
########################################################################################

class RegGrad(GradComputer):
    def __init__(self, net_params):
        super().__init__(net_params)
        
    def compute_grad(self, losses, param_offsets):
        losses = losses.cpu().detach().numpy()
        l0 = losses[0] # Loss at origin. Note that Noisy_BNN always samples first point at origin!
        grads = []
        for i in range(len(param_offsets)):
            smpls = param_offsets[i]
            S = smpls.shape[0]
            flat_noise = smpls.reshape((S, -1))
            flat_noise = flat_noise.cpu().detach().numpy()
            
            M, _, _, _ = np.linalg.lstsq(flat_noise, losses - l0)
            M = M.reshape(self.shapes[i])
            grads.append(torch.from_numpy(M).to(self.device))
        return grads
        
########################################################################################

########################################################################################
# Compute Gaussian smoothed gradients for the layers of model given loss samples.
# This is done by Monte-Carlo estimation where we approximate the gradient as a 
# weighted sum over the losses which are sampled from a normal distribution.
# d/dv_i of log(p_theta(v)) is (mu_i - v_i) / sigma_i^2, we scale by this in weighted sum.
########################################################################################

class SmoothGrad(GradComputer):
    def __init__(self, net_params, stddev):
        super().__init__(net_params)
        self.stddev = stddev
        
    def compute_grad(self, losses, param_offsets):
        grads = []
        S = losses.shape[0]
        for i in range(len(param_offsets)):
            # Sum 1/S * loss(v_i) * (mu_i - v_i) / sigma_i^2 
            smpls = param_offsets[i].reshape(S, -1)
            grad = torch.mean(losses.reshape(S, 1) * smpls, 0) / (self.stddev**2)
            grads.append(grad.reshape(self.shapes[i]))
        return grads

########################################################################################

########################################################################################
# Compute Gaussian smoothed LOSS, not gradients! This is very simply done 
# by Monte-Carlo estimation where we average the loss samples.
# Loss samples are thus assumed to be from normal distribution.
########################################################################################

def compute_smoothed_loss(losses):
    return torch.mean(losses)

########################################################################################
