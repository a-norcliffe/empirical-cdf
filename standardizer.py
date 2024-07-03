import torch


class Standardizer:
  """Standardizes torch tensors to be mean 0 and std 1."""
  def __init__(self):
    pass

  def fit(self, x, mask=None):
    if mask is None:
      self.mean = torch.mean(x, dim=0)
      self.std = torch.std(x, dim=0)
    else:
      self.mean = torch.sum(x*mask, dim=0) / torch.sum(mask, dim=0)
      # Bessel's correction.
      var = torch.sum(((x-self.mean)**2)*mask, dim=0)/(torch.sum(mask, dim=0)-1.0)
      self.std = var**0.5

  def transform(self, x, mask=None):
    with torch.no_grad():
      x = (x - self.mean) / self.std
      if mask is not None:
        x = x*mask
      return x