import torch

from torch.nn import functional as F
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss


class MultiLoss:
    def __init__(self, losses):
        self.losses = losses
        self.included = [
            CrossEntropyLoss
        ]  # Only the included ones apply the weight map

    def get_loss(self, loss, inp, target, weight_map):
        if any([type(loss) == x for x in self.included]):
            return (
                (loss(inp, target) * weight_map).mean()
                if weight_map is not None
                else loss(inp, target).mean()
            )
        return loss(inp, target).mean()

    def __call__(self, inp, target, weight_map=None):
        final_loss = None
        for loss in self.losses:
            if final_loss is None:
                final_loss = self.get_loss(loss, inp, target, weight_map)
            else:
                final_loss += self.get_loss(loss, inp, target, weight_map)
        return final_loss


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor.

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError(
            "inp labels type is not a torch.Tensor. Got {}".format(type(labels))
        )
    if not len(labels.shape) == 3:
        raise ValueError(
            "Invalid depth shape, we expect BxHxW. Got: {}".format(labels.shape)
        )
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype)
        )
    if num_classes < 1:
        raise ValueError(
            "The number of classes must be bigger than one."
            " Got: {}".format(num_classes)
        )
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(
        batch_size, num_classes, height, width, device=device, dtype=dtype
    )
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - inp: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> inp = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(inp, target)
        >>> output.backward()
    """

    def __init__(self, **kwargs) -> None:
        super(DiceLoss, self).__init__()
        self.eps: float = 1e-6

    def forward(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(inp):
            raise TypeError("inp type is not a torch.Tensor. Got {}".format(type(inp)))
        if not len(inp.shape) == 4:
            raise ValueError(
                "Invalid inp shape, we expect BxNxHxW. Got: {}".format(inp.shape)
            )
        if not inp.shape[-2:] == target.shape[-2:]:
            raise ValueError(
                "inp and target shapes must be the same. Got: {}".format(
                    inp.shape, inp.shape
                )
            )
        if not inp.device == target.device:
            raise ValueError(
                "inp and target must be in the same device. Got: {}".format(
                    inp.device, target.device
                )
            )
        # compute softmax over the classes axis
        inp_soft = F.softmax(inp, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(
            target, num_classes=inp.shape[1], device=inp.device, dtype=inp.dtype
        )

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(inp_soft * target_one_hot, dims)
        cardinality = torch.sum(inp_soft + target_one_hot, dims)

        dice_score = 2.0 * intersection / (cardinality + self.eps)
        return torch.mean(1.0 - dice_score)
