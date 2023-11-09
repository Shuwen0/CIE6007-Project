import torch
import torch.nn.functional as F

def binary_focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.25, reduction='mean'):
    """
    Compute binary focal loss between targets and logits.

    Args:
        logits (torch.Tensor): Logits from the model's output (shape: [batch_size] or [batch_size, 1])
        targets (torch.Tensor): Ground truth labels, binary (shape: [batch_size] or [batch_size, 1])
        gamma (float, optional): The focusing parameter gamma (default is 2.0).
        alpha (float, optional): Balanced scaling factor (default is 0.25).
        reduction (str, optional): The method used to reduce the loss ('none', 'mean', and 'sum').
    
    Returns:
        torch.Tensor: Calculated loss
    """
    # Apply the sigmoid function to clamp the logits to the [0,1] range
    probs = torch.sigmoid(logits)
    # Calculate the binary cross-entropy loss without reduction
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # Calculate the focal loss factors to modulate the BCE
    targets = targets.float()  # Ensure targets are floats (necessary if they're not already)
    focal_loss_factor = torch.where(targets == 1, 1 - probs, probs)
    focal_loss_factor = alpha * (focal_loss_factor ** gamma)
    # Apply focal loss factor to the BCE loss
    loss = focal_loss_factor * bce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# Example usage:
# logits = model(input) # shape: [batch_size, 1]
# targets = ... # shape: [batch_size, 1]
# loss = binary_focal_loss_with_logits(logits, targets)
