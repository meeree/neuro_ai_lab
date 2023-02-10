import torch

class FastSigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                S&≈\\frac{U}{1 + k|U|} \\\\
                \\frac{∂S}{∂U}&=\\frac{1}{(1+k|U|)^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.fast_sigmoid(slope=25)``.

    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning in
    Multilayer Spiking Neural Networks. Neural Computation, pp. 1514-1541.*"""
    @staticmethod
    def forward(ctx, input_, slope=25):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ > 1).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
        return grad, None

def fast_sigmoid(slope=25):
    """FastSigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return FastSigmoid.apply(x, slope)

    return inner
