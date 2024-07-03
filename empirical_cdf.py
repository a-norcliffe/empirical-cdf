import torch


class VectorEmpiricalCDF:
  """A torch vectorised empirical cdf calculator.
  We calculate by finding the locations of the quantiles, and then we
  linearly interpolate a general point. Finally we apply a Standard Normal
  inverse CDF to get the processed values.
  args:
    num_bins: int, the number of bins to use for the empirical cdf.
    size_normal: float, the standard deviation of the normal noise to add to the
                 data. This is added to prevent the cdf from being a step if
                 continuous values only fall into a finite set.
    ratio_uniform: float, the ratio of uniform noise to add to the data. We add
                  points sampled uniformly in the range of the data to prevent
                  the cdf from losing some of the information about gaps between
                  points.
  """
  def __init__(self, num_bins=200, size_normal=1e-5, ratio_uniform=0.1):
    self.num_bins = num_bins
    self.size_normal = size_normal
    self.ratio_uniform = ratio_uniform

  def linspace_batched(self, start, stop, steps):
    ints = (torch.arange(steps)).unsqueeze(0)
    dx = ((stop - start)/(steps - 1.0)).unsqueeze(-1)
    return start.unsqueeze(-1) + dx * ints

  def fit(self, x, mask=None):
    # If we have missing values we sample from the real data to fill them in.
    # This is acceptable since the CDFs are used to find quantiles, and each
    # feature is treated independently.
    # NOTE this assumes there are at least some non-missing values.
    if mask is not None:
      for f in range(x.shape[-1]):
        m_tmp = mask[:, f]
        real_x = x[:, f][torch.where(m_tmp)[0]]
        sampled_real_ids = torch.multinomial(torch.ones(real_x.shape[0]), x.shape[0], replacement=True)
        sampled_real_x = real_x[sampled_real_ids]
        x[:, f] = m_tmp*x[:, f] + (1-m_tmp)*sampled_real_x

    # Add normal noise.
    x = x + torch.randn(size=x.shape)*self.size_normal

    # Concatenate uniform data.
    min = torch.min(x, dim=0)[0]
    max = torch.max(x, dim=0)[0]
    uniform_data = (max-min)*torch.rand(size=(int(self.ratio_uniform*x.shape[0]), x.shape[-1])) + min
    x = torch.cat([x, uniform_data], dim=0)

    # Sort the data and find the quantiles.
    x = torch.sort(x, dim=0)[0]
    batchsize = x.shape[0]
    num_features = x.shape[1]

    # We start by looking at equally spaced quantiles. Equally spaced cdfs looks
    # in detail at regions where CDF changes quickly.
    # Step separates train set into that num_bins equally spaced bins.
    step = int(batchsize/self.num_bins)

    # These are the quantile values, i.e. the first has to be the minimum,
    # and the last has to be the maximum. Carried out in a batched way.
    self.s = torch.empty((num_features, self.num_bins))
    self.s[:, 0] = x[0, :]
    self.s[:, -1] = x[-1, :]

    # These are the cdf values at the quantile values, i.e. range from 0.0 to 1.0.
    self.cdf = torch.empty((num_features, self.num_bins))
    # This is an unbiased estimate of the min and max values drawn uniformly.
    # Rather than 0.0 and 1.0.
    self.cdf[:, 0] = 1/(batchsize+1)
    self.cdf[:, -1] = batchsize/(batchsize+1)

    for bin in range(1, self.num_bins - 1):
      self.s[:, bin] = x[step*bin, :]  # What x value is at this quantile.
      self.cdf[:, bin] = step*bin/batchsize  # What fraction of the data are we.

    # We then look at bins that are equally spaced in the x axis, this will
    # look in closer details at regions where the CDF changes slowly, which were
    # not previously captured.
    equally_spaced = self.linspace_batched(x[0, :], x[-1, :], self.num_bins)
    ids = torch.searchsorted((x.T).contiguous(), equally_spaced.contiguous(), right=True)
    cdf_x = ids/batchsize
    s_x = torch.gather(input=x.T, index=torch.clamp(ids, min=0, max=batchsize-1), dim=-1)
    s_x[:, 0] = x[0, :]
    s_x[:, -1] = x[-1, :]
    cdf_x[:, 0] = 1/(batchsize+1)
    cdf_x[:, -1] = batchsize/(batchsize+1)

    # Finally we merge the two, sort  and remove the first and last since they
    # are duplicates.
    self.s = torch.cat([self.s, s_x], dim=-1)
    self.cdf = torch.cat([self.cdf, cdf_x], dim=-1)
    self.s = torch.sort(self.s, dim=-1)[0][:, 1:-1]
    self.cdf = torch.sort(self.cdf, dim=-1)[0][:, 1:-1]

    # m is the constant in y = mx + c
    self.m = (self.cdf[:, 1:] - self.cdf[:, :-1]) / (self.s[:, 1:] - self.s[:, :-1])
    self.m = torch.cat([torch.zeros(self.m.shape[0], 1), self.m, torch.zeros(self.m.shape[0], 1)], dim=-1)

  def empirical_cdf(self, x):
    ids = torch.searchsorted(self.s.contiguous(), (x.T).contiguous(), right=True)
    m_ = torch.gather(input=self.m, index=ids, dim=-1).T
    # Shift ids back to correctly select the cdf and x_ values.
    clamped_ids = torch.clamp(ids-1, max=self.cdf.shape[-1]-1, min=0)
    c_ = torch.gather(input=self.cdf, index=clamped_ids, dim=-1).T
    x_ = torch.gather(input=self.s, index=clamped_ids, dim=-1).T
    return m_ * (x - x_) + c_

  def transform(self, x, mask=None):
    with torch.no_grad():
      x = torch.clamp(self.empirical_cdf(x), min=1e-7, max=1.0-1e-7)
      x = 2**0.5 * torch.erfinv(2*x - 1)
      if mask is not None:
        x = x*mask
      return x
