import torch
import importlib
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import element
import numpy as np
importlib.reload(element)

def camel(x, y):
  alpha = 0.2
  peak_1 = torch.exp(-(((x-1/4)**2 + (y-1/4)**2) / alpha**2))
  peak_2 = torch.exp(-(((x-3/4)**2 + (y-3/4)**2) / alpha**2))

  return (0.5 * (alpha * torch.sqrt(torch.tensor(torch.pi)))**(-2)) * (peak_1 + peak_2)


def plot_f_2d(f, N):
  xs, ys = torch.linspace(0, 1, N), torch.linspace(0, 1, N)
  X, Y = torch.meshgrid(xs, ys, indexing='ij')
  Z = f(X, Y)
  print(torch.var(Z))

  fig, ax = plt.subplots()

  pcm = plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')

  ax.set_xlabel("$x$", fontsize=14)
  ax.set_ylabel("$y$", fontsize=14)

  ax.tick_params(axis="both", which="major", direction="out", length=5, labelsize=10)
  ax.tick_params(axis="both", which="minor", direction="out", length=3)
  ax.minorticks_on()

  cbar = plt.colorbar(pcm, ax=ax)
  cbar.formatter = ticker.FormatStrFormatter('%.2f')
  cbar.update_ticks()

  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(0.0, 1.0)

  plt.show()


def grid_points_2d(n, density):

  npts = n
  grid = torch.empty(density*npts,2)

  for vi in range(int(density/2)):
    beg = vi*npts
    end = beg+npts
    grid[beg:end,0] = vi/int(((density/2) - 1))
    grid[beg:end,1] = torch.linspace(0, 1, npts)

  for hi in range(int(density/2)):
    beg = (int(density/2)+hi)*npts
    end = beg+npts
    grid[beg:end,1] = hi/int(((density/2)-1))
    grid[beg:end,0] = torch.linspace(0, 1, npts)

  return grid


def plot_f_scatter_2d(f, n, density):

  fig, ax = plt.subplots(figsize=(7.5,6))
  points = grid_points_2d(n, density)
  x = points[:,0]
  y = points[:,1]
  z = f(x, y)

  scat = ax.scatter(
    x,
    y,
    c=z,
    s=6,
    cmap='viridis'
  )
  ax.set_xlabel("$x$", fontsize=14)
  ax.set_ylabel("$y$", fontsize=14)

  ax.tick_params(axis="both", which="major", direction="out", length=5, labelsize=16)
  ax.tick_params(axis="both", which="minor", direction="out", length=3)
  ax.minorticks_on()

  cbar = plt.colorbar(scat, ax=ax)
  cbar.formatter = ticker.FormatStrFormatter('%.2f')
  cbar.ax.tick_params(labelsize=16)
  cbar.update_ticks()

  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(0.0, 1.0)

  plt.show()



def pwl_g_coupling(x_A, x_B, heights, bins):

  '''
  Coupling-layer compatible transform.  Preserves x_A, transforms x_B.
  New transform function generalised to deal with batches from NN.
  Parameters/bins can still be specified in the old way for testing by using an expanded tensor.
  x_A:      (B, D_A)
  x_B:      (B, D_B)
  D_A, D_B are the dimensionalities of the two partitions.  Both can be greater than 1.
  D_A, D_B pre-determined by specific mask of coupling layer.
  heights: (B, D_B, K-1)
  bins:   (B, D_B, K+1)
  '''

  device = x_B.device
  dtype = x_B.dtype

  B, D_B = x_B.shape
  K = heights.size(-1) + 1

  #cdf heights
  zeros = torch.zeros((B, D_B, 1), device=device, dtype=dtype)
  ones = torch.ones((B, D_B, 1), device=device, dtype=dtype)
  cdf_heights = torch.cat([zeros, heights, ones], dim=-1)

  #----------------APPLY TRANSFORM TO x_B-----------------------------------#
  # Compare x_B with bin edges

  xB_expanded = x_B.unsqueeze(-1)  # (B, D_B, 1)

  # Count how many bin edges x_B is greater than
  i_values = torch.sum(xB_expanded >= bins, dim=-1) - 1  # (B, D_B)
  i_values = torch.clamp(i_values, min=0, max=K-1)

  i_values_right = i_values + 1

  i_unsq = i_values.unsqueeze(-1)
  i_right_unsq = i_values_right.unsqueeze(-1)   #(B, D_B, 1)

  x_left  = torch.gather(bins, dim=2, index=i_unsq).squeeze(-1)   #(B, D_B, 1) ---> (B, D_B)
  x_right = torch.gather(bins, dim=2, index=i_right_unsq).squeeze(-1)

  y_left  = torch.gather(cdf_heights, dim=2, index=i_unsq).squeeze(-1)
  y_right = torch.gather(cdf_heights, dim=2, index=i_right_unsq).squeeze(-1)    #(B, D_B, 1) ---> (B, D_B)

  slope = (y_right - y_left) / (x_right - x_left)
  y_B = y_left + slope * (x_B - x_left)

  #------------------------------------------------------------------------#

  y_A = x_A

  return y_A, y_B


def pwl_expand_params(B, params):   #copies parameters to fit batch size
  '''
  params.shape = (D, K-1)
  returns (B, D, K-1)
  '''

  D = params.size(0)
  K = params.size(1) + 1
  expanded_params = torch.empty(B, D, K-1)
  expanded_params[:, :] = params
  return expanded_params

def expand_edges(B, edges):   #copies bin edges to fit batch size
  '''
  edges.shape = (D, K+1)
  returns (B, D, K+1)
  '''

  D = edges.size(0)
  K = edges.size(1) -1
  expanded_edges = torch.empty(B, D, K+1)
  expanded_edges[:, :] = edges
  return expanded_edges


def pwl_raw_heights_to_params(raw_heights):
    """
    raw_heights: (B, D, K-1)

    Returns:
        params: (B, D, K-1)
        strictly increasing in last dimension,
        between 0 and 1.
    """

    device = raw_heights.device
    dtype = raw_heights.dtype

    B, D, K_minus_1 = raw_heights.shape

    # Append extra zero for normalization
    zeros = torch.zeros((B, D, 1), device=device, dtype=dtype)

    u_ext = torch.cat([raw_heights, zeros], dim=-1)  # (B, D, K)

    # Softmax along bin dimension
    w = torch.softmax(u_ext, dim=-1)  # (B, D, K)

    # Cumulative sum to get CDF
    ys = torch.cumsum(w, dim=-1)  # (B, D, K)

    # Remove last entry (which equals 1)
    return ys[..., :-1]  # (B, D, K-1)


def pwl_raw_heights_to_params_stable(raw_heights, min_cdf_inc=1e-3):
    """
    raw_heights: (B, D, K-1)

    returns heights: (B, D, K-1)
    such that cdf increments are all >= min_cdf_inc
    """

    device = raw_heights.device
    dtype = raw_heights.dtype

    B, D, K_minus_1 = raw_heights.shape
    K = K_minus_1 + 1

    if not (0.0 <= min_cdf_inc < 1.0 / K):
        raise ValueError(f"Need 0 <= min_cdf_inc < 1/K, got {min_cdf_inc} with K={K}")

    zeros = torch.zeros((B, D, 1), device=device, dtype=dtype)
    u_ext = torch.cat([raw_heights, zeros], dim=-1)   # (B, D, K)

    incs_soft = torch.softmax(u_ext, dim=-1)
    incs = min_cdf_inc + (1.0 - K * min_cdf_inc) * incs_soft

    cdf = torch.cumsum(incs, dim=-1)                  # (B, D, K)
    return cdf[..., :-1]                              # interior heights



def raw_widths_to_bins(raw_widths):
    """
    raw_widths: (B, D, K)

    Returns:
        bins: (B, D, K+1)
        increasing,
        starting at 0 and ending at 1.
    """

    device = raw_widths.device
    dtype = raw_widths.dtype

    # Softmax so widths sum to 1
    widths = torch.softmax(raw_widths, dim=-1)  # (B, D, K)

    # Cumulative sum gives interior edges
    int_edges = torch.cumsum(widths, dim=-1)  # (B, D, K)

    # Prepend zero
    zeros = torch.zeros((*int_edges.shape[:2], 1),
                        device=device, dtype=dtype)

    bins = torch.cat([zeros, int_edges], dim=-1)  # (B, D, K+1)

    return bins

def raw_widths_to_bins_stable(raw_widths, min_bin_width=1e-3):
    """
    raw_widths: (B, D, K)
    returns bins: (B, D, K+1)
    """

    device = raw_widths.device
    dtype = raw_widths.dtype
    K = raw_widths.size(-1)

    if not (0.0 <= min_bin_width < 1.0 / K):
        raise ValueError(f"Need 0 <= min_bin_width < 1/K, got {min_bin_width} with K={K}")

    widths_soft = torch.softmax(raw_widths, dim=-1)
    widths = min_bin_width + (1.0 - K * min_bin_width) * widths_soft    #every bin has at least min_bin_width and there's K bins so that leaves (1-K)min_bin_width to control

    zeros = torch.zeros((*widths.shape[:2], 1), device=device, dtype=dtype)
    bins = torch.cat([zeros, torch.cumsum(widths, dim=-1)], dim=-1)

    # optional: enforce exact right endpoint 1.0
    bins[..., -1] = 1.0
    return bins


def pwl_inverse_transform(y_B, B_dims, heights, bins):

  '''
  Computes the inverse transform and associated Jacobian.
  Non-uniform bins permitted.
  y_B:      (B, D_B)
  '''

  device = y_B.device
  dtype = y_B.dtype

  B, D_B = y_B.shape
  K = heights.size(-1) + 1

  #cdf heights
  zeros = torch.zeros((B, D_B, 1), device=device, dtype=dtype)
  ones = torch.ones((B, D_B, 1), device=device, dtype=dtype)
  cdf_heights = torch.cat([zeros, heights, ones], dim=-1)

  #---------y_A = x_A so only need to do inverse for y_B--------------------#

  #sort y_B by comparing y_B with cdf heights
  yB_expanded = y_B.unsqueeze(-1)  # (B, D_B, 1)

  j_values = torch.sum(yB_expanded >= cdf_heights, dim=-1) - 1  #(B, D_B)
  j_values = torch.clamp(j_values, min=0, max=K-1)

  j_values_right = j_values + 1

  j_unsq = j_values.unsqueeze(-1)
  j_right_unsq = j_values_right.unsqueeze(-1)

  x_left = torch.gather(bins, dim=2, index=j_unsq).squeeze(-1)    #(B, D_B, 1) ---> (B, D_B)
  x_right = torch.gather(bins, dim=2, index=j_right_unsq).squeeze(-1)
  y_left = torch.gather(cdf_heights, dim=2, index=j_unsq).squeeze(-1)
  y_right = torch.gather(cdf_heights, dim=2, index=j_right_unsq).squeeze(-1)

  def g_inv(y):

    return x_left + ((x_right - x_left) / (y_right - y_left)) * (y - y_left)    #(B, D_B)

  jacobians = (x_right - x_left) / (y_right - y_left)   #(B, D_B)

  jac_det = jacobians.prod(dim=1)

  return g_inv(y_B), jac_det


#neural network for cdf heights V

class flownet_V(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):

    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),   #only one hidden layer for now
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=output_size),
    )

  def forward(self, x):
        return self.linear_layer_stack(x)



#neural network for bin widths W

class flownet_W(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):

    super().__init__()
    self.linear_layer_stack = nn.Sequential(
        nn.Linear(in_features=input_size, out_features=hidden_size),
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=hidden_size),   #only one hidden layer for now
        nn.ReLU(),
        nn.Linear(in_features=hidden_size, out_features=output_size),
    )

  def forward(self, x):
        return self.linear_layer_stack(x)


class pwl_layer(nn.Module):
  '''
  A self-contained piecewise-linear coupling layer.
  Includes forward transform, inverse transform and Jacobian determinant functions.
  '''
  def __init__(self, D_A, D_B, A_dims, B_dims, K, hidden_size, min_bin_width=1e-3, min_cdf_inc=1e-3):

    super().__init__()

    #attributes

    self.D_A = D_A
    self.D_B = D_B
    self.K = K
    self.min_bin_width = min_bin_width
    self.min_cdf_inc = min_cdf_inc
    self.register_buffer("A_dims", A_dims)
    self.register_buffer("B_dims", B_dims)

    #networks

    self.heights_net = flownet_V(input_size=D_A, hidden_size=hidden_size, output_size=D_B*(K-1))

    self.widths_net = flownet_W(input_size=D_A, hidden_size=hidden_size, output_size=D_B*K)

  def _heights_from_raw(self, raw_heights):
    return pwl_raw_heights_to_params_stable(
        raw_heights,
        min_cdf_inc=self.min_cdf_inc
    )

  def _bins_from_raw(self, raw_widths):
    return raw_widths_to_bins_stable(
        raw_widths,
        min_bin_width=self.min_bin_width
    )


  def forward(self, x):

    batch = x.size(0)

    x_A = x[:, self.A_dims]   #apply mask
    x_B = x[:, self.B_dims]

    A = self.A_dims.tolist()
    B = self.B_dims.tolist()
    D_total = x.size(1)

    assert len(A) == len(set(A)), f"Duplicate entries in A_dims: {A}"
    assert len(B) == len(set(B)), f"Duplicate entries in B_dims: {B}"

    Aset = set(A)
    Bset = set(B)
    full = set(range(D_total))

    assert Aset.isdisjoint(Bset), f"Overlap in masks: {Aset & Bset}"
    assert (Aset | Bset) == full, f"Missing dims: {full - (Aset | Bset)}"

    raw_heights = self.heights_net(x_A).reshape(batch, self.D_B, (self.K - 1))    #evaluate neural networks with x_A
    raw_widths = self.widths_net(x_A).reshape(batch, self.D_B, self.K)

    #heights = self._heights_from_raw(raw_heights)
    heights = pwl_raw_heights_to_params(raw_heights)
    #bins = self._bins_from_raw(raw_widths)
    bins = raw_widths_to_bins(raw_widths)

    y_A, y_B = pwl_g_coupling(x_A, x_B, heights, bins)    #apply the transform to x_B

    #reconstruct the y vector

    y = x.clone()
    #y[:, self.A_dims] = y_A
    y[:, self.B_dims] = y_B

    return y


  def inverse(self, y):

    batch = y.size(0)

    y_A = y[:, self.A_dims]
    y_B = y[:, self.B_dims]

    A = self.A_dims.tolist()
    B = self.B_dims.tolist()
    D_total = y.size(1)

    assert len(A) == len(set(A)), f"Duplicate entries in A_dims: {A}"
    assert len(B) == len(set(B)), f"Duplicate entries in B_dims: {B}"

    Aset = set(A)
    Bset = set(B)
    full = set(range(D_total))

    assert Aset.isdisjoint(Bset), f"Overlap in masks: {Aset & Bset}"
    assert (Aset | Bset) == full, f"Missing dims: {full - (Aset | Bset)}"

    raw_heights = self.heights_net(y_A).reshape(batch, self.D_B, (self.K - 1))    #evaluate neural networks with y_A
    raw_widths = self.widths_net(y_A).reshape(batch, self.D_B, self.K)

    #heights = pwl_raw_heights_to_params_stable(raw_heights, min_cdf_inc=self.min_cdf_inc)
    #bins = raw_widths_to_bins_stable(raw_widths, min_bin_width=self.min_bin_width)
    heights = pwl_raw_heights_to_params(raw_heights)
    bins = raw_widths_to_bins(raw_widths)

    x_B, jac_det = pwl_inverse_transform(y_B, self.B_dims, heights, bins)
    x_A = y_A

    x = y.clone()
    #x[:, self.A_dims] = x_A
    x[:, self.B_dims] = x_B

    return x, jac_det


class Composition(nn.Module):
  '''
  Composes the specified coupling layers and has the same functions as its constitients but generalised.
  '''

  def __init__(self, layers):

    super().__init__()

    self.layers = nn.ModuleList(layers)

  def forward(self, x):

    Y = x

    for layer in self.layers:

      Y = layer(Y)

    return Y

  def inverse(self, y):

    #we need to iterate backwards

    X = y
    jac_dets = torch.ones(y.shape[0], device=y.device, dtype=y.dtype)

    for layer in reversed(list(self.layers)):

      X, jac = layer.inverse(X)
      jac_dets = jac_dets * jac

    return X, jac_dets
  


def pwq_expand_heights(B, heights):   #copies parameters to fit batch size
  '''
  heights.shape = (D, K+1)
  returns (B, D, K+1)
  '''

  D = heights.size(0)
  K = heights.size(1) - 1
  expanded_heights = torch.empty(B, D, K+1)
  expanded_heights[:, :] = heights
  return expanded_heights


def preprocess_params_2(heights, bins):
    """
    heights: (B, D_B, K+1) unnormalised PDF heights
    bins:    (B, D_B, K+1) bin edges - ascending; sum to 1
    returns:
      v:     (B, D_B, K+1) normalised PDF heights
      bin_areas: (B, D_B, K)   normalised bin areas
    """

    v = heights
    v_left = v[:, :, :-1]
    v_right = v[:, :, 1:]
    bin_widths = bins[:, :, 1:] - bins[:, :, :-1]

    bin_areas = 0.5 * (v_left + v_right) * bin_widths
    total_area = bin_areas.sum(dim=-1, keepdim=True)

    # normalise
    v = v / total_area
    bin_areas = bin_areas / total_area

    return v, bin_areas


def pwq_g_coupling(x_A, x_B, heights, bins):

  '''
  Piecewise quadratic transform with K bins, designed for coupling layers.
  Preserves x_A, transforms x_B.
  x_A:    (B, D_A)
  x_B:    (B, D_B)
  D_A, D_B are the dimensionalities of the two partitions.  Both can be greater than 1.
  D_A, D_B pre-determined by specific mask of coupling layer.
  heights: (B, D_B, K+1)
  bins:    (B, D_B, K+1)

  '''

  device = x_B.device
  dtype = x_B.dtype

  B, D_B = x_B.shape
  K = heights.size(-1) - 1

  #----------------APPLY TRANSFORM TO x_B-----------------------------------#
  # Compare x_B with bin edges

  xB_expanded = x_B.unsqueeze(-1)  # (B, D_B, 1)

  # Count how many bin edges x_B is greater than
  i_values = torch.sum(xB_expanded >= bins, dim=-1) - 1  # (B, D_B)
  i_values = torch.clamp(i_values, min=0, max=K-1)

  i_values_right = i_values + 1

  v, bin_areas = preprocess_params_2(heights, bins)  #get normalised pdf heights and bin areas

  i_unsq = i_values.unsqueeze(-1)
  i_right_unsq = i_values_right.unsqueeze(-1)   #(B, D_B, 1)

  x_left  = torch.gather(bins, dim=2, index=i_unsq).squeeze(-1)   #(B, D_B, 1) ---> (B, D_B)
  x_right = torch.gather(bins, dim=2, index=i_right_unsq).squeeze(-1)
  w_b = x_right - x_left

  #pdf heights
  v_left  = torch.gather(v, dim=2, index=i_unsq).squeeze(-1)
  v_right = torch.gather(v, dim=2, index=i_right_unsq).squeeze(-1)    #(B, D_B, 1) ---> (B, D_B)
  w_v = v_right - v_left    #slope of pdf inside bin (linear)

  #transform to local coordinate alpha
  eps = torch.finfo(dtype).eps
  alpha = (x_B - x_left) / (w_b + eps)    #avoid instability;  (B, D_B)

  #cumulative area to left of x values
  cum_areas = torch.zeros(B, D_B, device=device, dtype=dtype)
  for d in range(D_B):
    for i in range(B):
      cum_areas[i, d] = torch.sum(bin_areas[i, d, :i_values[i, d]])

  c_left = cum_areas

  #compute quadratic mappings
  y_B = c_left + ((alpha**2/2) * w_v * w_b) + (alpha * v_left * w_b)

  #------------------------------------------------------------------------#

  y_A = x_A

  return y_A, y_B


def pwq_inverse_transform(y_B, B_dims, heights, bins):
  '''
  Computes the inverse transform and associated Jacobian.
  Non-uniform bins permitted.
  y_B:      (B, D_B)
  '''

  device = y_B.device
  dtype = y_B.dtype

  B, D_B = y_B.shape
  K = heights.size(-1) - 1

  v, bin_areas = preprocess_params_2(heights, bins)

  #---------y_A = x_A so only need to do inverse for y_B--------------------#

  #cumulative areas

  c_edges = torch.zeros((B, D_B, K+1), device=device, dtype=dtype)   #cdf starts at 0
  c_edges[:, :, 1:] = torch.cumsum(bin_areas, dim=-1)

  #find indices; sort y_B by comparing with c_edges

  y_B_expanded = y_B.unsqueeze(-1)  # (B, D_B, 1)

  j_values = torch.sum(y_B_expanded >= c_edges, dim=-1) - 1 #(B, D_B)
  j_values = torch.clamp(j_values, min=0, max=K-1)

  j_values_right = j_values + 1

  j_unsq = j_values.unsqueeze(-1)
  j_right_unsq = j_values_right.unsqueeze(-1)

  #compute the edge positions

  x_left = torch.gather(bins, dim=2, index=j_unsq).squeeze(-1)    #(B, D_B, 1) ---> (B, D_B)
  x_right = torch.gather(bins, dim=2, index=j_right_unsq).squeeze(-1)
  w_b = x_right - x_left    #bin width(s)

  c_left = torch.gather(c_edges, dim=2, index=j_unsq).squeeze(-1)

  #pdf heights

  v_left = torch.gather(v, dim=2, index=j_unsq).squeeze(-1)
  v_right = torch.gather(v, dim=2, index=j_right_unsq).squeeze(-1)
  w_v = v_right - v_left    #vertex differences

  #solve quadratic to obtain alpha values

  # coefficients
  a = 0.5 * w_v * w_b
  b = v_left * w_b
  c = c_left - y_B

  # linear vs quadratic bins
  eps = torch.finfo(dtype).eps
  is_linear = torch.abs(w_v) < eps

  alpha = torch.empty_like(y_B, device=device, dtype=dtype)   #(B, D_B)

  # linear case
  alpha[is_linear] = (y_B[is_linear] - c_left[is_linear]) / (v_left[is_linear] * w_b[is_linear])

  # quadratic case
  disc = torch.clamp(b*b - 4*a*c, min=0.0)
  alpha[~is_linear] = (-b[~is_linear] + torch.sqrt(disc[~is_linear])) / (2*a[~is_linear])

  #jacobians

  jac = 1 / (v_left + (w_v*alpha))    #(B, D_B)
  jac_det = jac.prod(dim=1)

  #obtain x values
  x_B = x_left + (w_b*alpha)

  #-------------------------------------------------------------------------#

  return x_B, jac_det


class pwq_layer(nn.Module):
  '''
  A self-contained piecewise-quadratic coupling layer.
  Includes forward transform, inverse transform and Jacobian determinant functions.
  '''

  def __init__(self, D_A, D_B, A_dims, B_dims, K, hidden_size, min_bin_width=1e-3):

    super().__init__()

    #attributes

    self.D_A = D_A
    self.D_B = D_B
    self.K = K
    self.register_buffer("A_dims", A_dims)
    self.register_buffer("B_dims", B_dims)
    self.min_bin_width = min_bin_width

    #networks

    self.heights_net = flownet_V(input_size=D_A, hidden_size=hidden_size, output_size=(D_B * (K+1)))

    self.widths_net = flownet_W(input_size=D_A, hidden_size=hidden_size, output_size=(D_B * K))

  def _bins_from_raw(self, raw_widths):
    return raw_widths_to_bins_stable(
        raw_widths,
        min_bin_width=self.min_bin_width
    )

  
  def forward(self, x):

    batch = x.size(0)

    x_A = x[:, self.A_dims]   #apply mask
    x_B = x[:, self.B_dims]

    A = self.A_dims.tolist()
    B = self.B_dims.tolist()
    D_total = x.size(1)

    assert len(A) == len(set(A)), f"Duplicate entries in A_dims: {A}"
    assert len(B) == len(set(B)), f"Duplicate entries in B_dims: {B}"

    Aset = set(A)
    Bset = set(B)
    full = set(range(D_total))

    assert Aset.isdisjoint(Bset), f"Overlap in masks: {Aset & Bset}"
    assert (Aset | Bset) == full, f"Missing dims: {full - (Aset | Bset)}"

    raw_heights = self.heights_net(x_A).reshape(batch, self.D_B, (self.K+1))
    heights = torch.nn.functional.softplus(raw_heights) + torch.finfo(x.dtype).eps # Ensure positivity
    raw_widths = self.widths_net(x_A).reshape(batch, self.D_B, self.K)

    #bins = self._bins_from_raw(raw_widths)
    bins = raw_widths_to_bins(raw_widths)

    y_A, y_B = pwq_g_coupling(x_A, x_B, heights, bins)    #apply transform to x_B

    #reconstruct the y vector

    y = x.clone()
    #y[:, self.A_dims] = y_A
    y[:, self.B_dims] = y_B

    return y


  def inverse(self, y):

    batch = y.size(0)

    y_A = y[:, self.A_dims]
    y_B = y[:, self.B_dims]

    A = self.A_dims.tolist()
    B = self.B_dims.tolist()
    D_total = y.size(1)

    assert len(A) == len(set(A)), f"Duplicate entries in A_dims: {A}"
    assert len(B) == len(set(B)), f"Duplicate entries in B_dims: {B}"

    Aset = set(A)
    Bset = set(B)
    full = set(range(D_total))

    assert Aset.isdisjoint(Bset), f"Overlap in masks: {Aset & Bset}"
    assert (Aset | Bset) == full, f"Missing dims: {full - (Aset | Bset)}"

    raw_heights = self.heights_net(y_A).reshape(batch, self.D_B, (self.K+1))
    heights = torch.nn.functional.softplus(raw_heights) + torch.finfo(y.dtype).eps # Ensure positivity
    raw_widths = self.widths_net(y_A).reshape(batch, self.D_B, self.K)

    #bins = self._bins_from_raw(raw_widths)
    bins = raw_widths_to_bins(raw_widths)
    x_B, jac_det = pwq_inverse_transform(y_B, self.B_dims, heights, bins)
    x_A = y_A

    x = y.clone()
    #x[:, self.A_dims] = x_A
    x[:, self.B_dims] = x_B

    return x, jac_det



def normalising_flow(layer_type, A_dims, B_dims, K, hidden_size, min_bin_width=1e-3, min_cdf_inc=1e-3):

  '''
  Works for arbitrary dimensions, masks, layers.
  A_dims, B_dims are lists of tensors containing the indices of the relevant dimensions - the maskings.
  K is the number of bins for the piecewise-linear transformation.
  hidden_size is the number of nodes in the hidden layer(s) of the neural networks.
  Returns the Composition object corresponding to the inputs.
  '''

  num_layers = len(A_dims)
  layers = []

  for i in range(num_layers):   #build layers

    D_A = A_dims[i].size(-1)    #get D_A, D_B
    D_B = B_dims[i].size(-1)

    if layer_type == pwl_layer:
    

      layer = pwl_layer(
          D_A = D_A,
          D_B = D_B,
          A_dims = A_dims[i],
          B_dims = B_dims[i],
          K = K,                  #using same K for each layer for now
          hidden_size = hidden_size,
          min_bin_width = min_bin_width,
          min_cdf_inc=min_cdf_inc,
      )

    else:
       
       layer = pwq_layer(
          D_A = D_A,
          D_B = D_B,
          A_dims = A_dims[i],
          B_dims = B_dims[i],
          K = K,                  #using same K for each layer for now
          hidden_size = hidden_size,
          min_bin_width = min_bin_width
       )
    
    layers.append(layer)


  c = Composition(layers)
  return c


def train_loop_complete(f, D, layer_type, A_dims, B_dims, K, hidden_size, N, epochs, lr, device, ticker):    #using same K for each layer for now

  #instantiate flow with specified parameters
  flow = normalising_flow(
      layer_type,
      A_dims=A_dims,
      B_dims=B_dims,
      K=K,
      hidden_size=hidden_size
  )

  #optimiser
  optimiser = torch.optim.Adam([
      {"params": flow.parameters(), "lr": lr}
      ])

  loss_values = torch.zeros(epochs, device=device)

  for epoch in range(epochs):

    V = torch.rand(N, D)

    #inverse, jacobian determinants
    X, jac_dets = flow.inverse(V)

    #compute new function
    X = X.T
    h_eval = f(*X) * torch.abs(jac_dets)

    #compute loss
    loss = torch.var(h_eval)
    loss_values[epoch] = loss.detach()

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    if ticker and epoch % 20 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.8e}")
      print("Mean value: ", torch.mean(h_eval).detach().item())

  final_loss = loss.detach()

  return final_loss, loss_values, flow


def train_loop_decay(D, layer_type, A_dims, B_dims, K, hidden_size, N, epochs, lr, device, dtype, ticker):    #using same K for each layer for now

  if dtype is np.float64:
    torch_dtype = torch.float64
  elif dtype is np.float32:
    torch_dtype = torch.float32
  else:
    torch_dtype=dtype

  #instantiate flow with specified parameters
  flow = normalising_flow(
      layer_type,
      A_dims=A_dims,
      B_dims=B_dims,
      K=K,
      hidden_size=hidden_size
  ).to(device=device, dtype=torch_dtype)

  #optimiser
  optimiser = torch.optim.Adam([
      {"params": flow.parameters(), "lr": lr}
      ])
  
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=[5500], gamma=0.3)

  #normalisation - determine scaling from initial batch evaluation

  with torch.no_grad():
    V_scale = torch.rand((5*N, D), device=device, dtype=torch_dtype)    #large batch
    X_scale, jac_dets_scale = flow.inverse(V_scale)
    (P, P1, P2, P3), jac_map_scale = element.hypercube_to_momenta(X_scale, m_t=173)
    me2_scale = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)
    h_scale = me2_scale * jac_dets_scale * jac_map_scale
    scale = torch.std(h_scale)
    print('Normalisation scale:', scale.item())


  loss_values = torch.zeros(epochs, device=device, dtype=torch_dtype)

  for epoch in range(epochs):

    V = torch.rand((N, D), device=device, dtype=torch_dtype)    #[0, 1]^5

    #inverse, jacobian determinants
    X, jac_dets = flow.inverse(V)   #X still [0,1]^5

    #map transformed points to usable variables
    #[0, 1]^5 --> E1, E2, costheta1, phi1, phi2
    #get N sets of momenta from the N sets of E1, E2, ...
    (P, P1, P2, P3), jac_map = element.hypercube_to_momenta(X, m_t=173)

    #evaluate N matrix elements

    me2 = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)

    h_evals = me2 * jac_dets * jac_map
    #h_evals = jac_dets * jac_map
    #normalise

    h_norm = h_evals / scale

    #compute loss
    loss = torch.var(h_norm)
    loss_values[epoch] = loss.detach()

    #scheduler.step(loss.detach())

    optimiser.zero_grad()

    loss.backward()

    #torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

    optimiser.step()

    if ticker and epoch % 20 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.8e}")
      print("Mean value: ", torch.mean(h_evals).detach().item())
      print("Std h :", torch.std(h_evals).detach().item())
      #print("me2 min/max:", me2.min().item(), me2.max().item())
      #print("jac_dets min/max:", jac_dets.min().item(), jac_dets.max().item())
      #print("jac_dets abs mean:", jac_dets.abs().mean().item())
      #print("X min/max:", X.min().item(), X.max().item())
      #print("X, jac_dets, me2, h_evals all finite?",
            #torch.isfinite(X).all().item(),
            #torch.isfinite(jac_dets).all().item(),
            #torch.isfinite(me2).all().item(),
            #torch.isfinite(h_evals).all().item())

  final_loss = loss.detach()

  return final_loss, loss_values, flow





def _as_torch_dtype(dtype):
    if dtype is np.float64:
        return torch.float64
    if dtype is np.float32:
        return torch.float32
    return dtype


def _tensor_stats(name, x):
    x_flat = x.reshape(-1)
    finite = torch.isfinite(x_flat)
    n_total = x_flat.numel()
    n_finite = finite.sum().item()
    frac_finite = n_finite / n_total if n_total > 0 else 1.0

    out = {
        f"{name}_n": n_total,
        f"{name}_finite_frac": frac_finite,
    }

    if n_finite == 0:
        out.update({
            f"{name}_mean": float("nan"),
            f"{name}_std": float("nan"),
            f"{name}_min": float("nan"),
            f"{name}_max": float("nan"),
            f"{name}_abs_mean": float("nan"),
        })
        return out

    xf = x_flat[finite]
    out.update({
        f"{name}_mean": xf.mean().item(),
        f"{name}_std": xf.std(unbiased=False).item(),
        f"{name}_min": xf.min().item(),
        f"{name}_max": xf.max().item(),
        f"{name}_abs_mean": xf.abs().mean().item(),
    })
    return out


def _grad_and_param_norm(flow):
    grad_sq = 0.0
    param_sq = 0.0
    max_grad_abs = 0.0
    n_grad_finite = 0
    n_grad_total = 0

    for p in flow.parameters():
        param_sq += p.detach().pow(2).sum().item()

        if p.grad is not None:
            g = p.grad.detach()
            finite = torch.isfinite(g)
            n_grad_total += g.numel()
            n_grad_finite += finite.sum().item()

            if finite.any():
                gf = g[finite]
                grad_sq += gf.pow(2).sum().item()
                max_grad_abs = max(max_grad_abs, gf.abs().max().item())

    grad_norm = grad_sq ** 0.5
    param_norm = param_sq ** 0.5
    grad_finite_frac = (
        n_grad_finite / n_grad_total if n_grad_total > 0 else 1.0
    )

    return {
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "max_grad_abs": max_grad_abs,
        "grad_finite_frac": grad_finite_frac,
    }


@torch.inference_mode()
def estimate_unit_integral(flow, D, n_test=200_000, batch_size=50_000, device=None, dtype=None):
    if device is None:
        device = next(flow.parameters()).device
    if dtype is None:
        dtype = next(flow.parameters()).dtype

    sum_jac = torch.tensor(0.0, device=device, dtype=dtype)
    sum_jac2 = torch.tensor(0.0, device=device, dtype=dtype)
    n_done = 0

    for start in range(0, n_test, batch_size):
        b = min(batch_size, n_test - start)
        Y = torch.rand((b, D), device=device, dtype=dtype)
        _, jac = flow.inverse(Y)

        jac = jac.reshape(-1)
        finite = torch.isfinite(jac)
        jac_f = jac[finite]

        if jac_f.numel() > 0:
            sum_jac += jac_f.sum()
            sum_jac2 += (jac_f ** 2).sum()
            n_done += jac_f.numel()

    if n_done == 0:
        return {
            "unit_int_mean": float("nan"),
            "unit_int_error": float("nan"),
            "unit_int_n": 0,
        }

    mean = sum_jac / n_done
    var = sum_jac2 / n_done - mean ** 2
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)
    err = std / torch.sqrt(torch.tensor(n_done, device=device, dtype=dtype))

    return {
        "unit_int_mean": mean.item(),
        "unit_int_error": err.item(),
        "unit_int_n": n_done,
    }

def train_loop_decay_diagnostic(
    D,
    layer_type,
    A_dims,
    B_dims,
    K,
    hidden_size,
    N,
    epochs,
    lr,
    device,
    dtype,
    ticker=True,
    print_every=20,
    unit_test_every=100,
    unit_test_samples=500_000,
    clip_grad=None,
    m_t=173,
):
    torch_dtype = _as_torch_dtype(dtype)

    flow = normalising_flow(
        layer_type=layer_type,
        A_dims=A_dims,
        B_dims=B_dims,
        K=K,
        hidden_size=hidden_size,
        min_bin_width=0.001,
        min_cdf_inc = 0.001
    ).to(device=device, dtype=torch_dtype)

    optimiser = torch.optim.Adam(flow.parameters(), lr=lr)

    with torch.no_grad():
        V_scale = torch.rand((5 * N, D), device=device, dtype=torch_dtype)
        X_scale, jac_dets_scale = flow.inverse(V_scale)
        (P, P1, P2, P3), jac_map_scale = element.hypercube_to_momenta(X_scale, m_t=m_t)
        me2_scale = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)
        h_scale = me2_scale * jac_dets_scale * jac_map_scale
        scale = torch.std(h_scale)

    print(f"Initial normalisation scale: {scale.item():.6e}")

    loss_values = torch.zeros(epochs, device=device, dtype=torch_dtype)
    diagnostics = []

    for epoch in range(epochs):
        V = torch.rand((N, D), device=device, dtype=torch_dtype)

        X, jac_dets = flow.inverse(V)
        (P, P1, P2, P3), jac_map = element.hypercube_to_momenta(X, m_t=m_t)
        me2 = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)

        h_evals = me2 * jac_dets * jac_map
        h_norm = h_evals / scale

        loss = torch.var(h_norm)
        loss_values[epoch] = loss.detach()

        optimiser.zero_grad()
        loss.backward()

        grad_info_preclip = _grad_and_param_norm(flow)

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=clip_grad)

        grad_info_postclip = _grad_and_param_norm(flow)

        optimiser.step()

        with torch.inference_mode():
            #x_oob_frac = ((X < 0) | (X > 1)).double().mean().item()

            record = {
                "epoch": epoch,
                "loss": loss.detach().item(),
                "scale": scale.item(),
                #"x_oob_frac": x_oob_frac,
            }
            record.update(_tensor_stats("jac", jac_dets))
            record.update(_tensor_stats("jac_map", jac_map))
            record.update(_tensor_stats("me2", me2))
            record.update(_tensor_stats("h", h_evals))
            record.update(_tensor_stats("h_norm", h_norm))
            record.update(_tensor_stats("x", X))
            record.update({f"preclip_{k}": v for k, v in grad_info_preclip.items()})
            record.update({f"postclip_{k}": v for k, v in grad_info_postclip.items()})

            if unit_test_every is not None and (epoch % unit_test_every == 0 or epoch == epochs - 1):
                record.update(
                    estimate_unit_integral(
                        flow, D, n_test=unit_test_samples, batch_size=min(50_000, unit_test_samples),
                        device=device, dtype=torch_dtype
                    )
                )

            diagnostics.append(record)

        if ticker and (epoch % print_every == 0 or epoch == epochs - 1):
            msg = (
                f"Epoch {epoch:5d} | "
                f"loss={record['loss']:.6e} | "
                f"jac_mean={record['jac_mean']:.6e} | "
                f"jac_std={record['jac_std']:.6e} | "
                f"jac_min={record['jac_min']:.6e} | "
                f"jac_max={record['jac_max']:.6e} | "
                f"jac_map_mean={record['jac_map_mean']:.6e} | "
                #f"me2_mean={record['me2_mean']:.6e} | "
                #f"h_mean={record['h_mean']:.6e} | "
                f"h_std={record['h_std']:.6e} | "
                #f"x_oob={record['x_oob_frac']:.3e} | "
                f"grad_norm={record['postclip_grad_norm']:.6e}"
            )
            if "unit_int_mean" in record:
                msg += (
                    f" | unit_int={record['unit_int_mean']:.6e}"
                    f" ± {record['unit_int_error']:.2e}"
                )
            print(msg)

            if (
                record["jac_finite_frac"] < 1.0
                or record["x_finite_frac"] < 1.0
                or record["jac_map_finite_frac"] < 1.0
                or record["me2_finite_frac"] < 1.0
                or record["h_finite_frac"] < 1.0
            ):
                print("  WARNING: non-finite values detected.")

    final_loss = loss.detach()
    return final_loss, loss_values, flow, diagnostics



def _geometry_summary_across_layers(layer_summaries):
    """
    Reduce per-layer geometry summaries to a compact set of global diagnostics.
    """
    if len(layer_summaries) == 0:
        return {}

    return {
        "geom_width_min_global": min(s["width_min"] for s in layer_summaries),
        "geom_width_max_global": max(s["width_max"] for s in layer_summaries),
        "geom_cdf_inc_min_global": min(s["cdf_inc_min"] for s in layer_summaries),
        "geom_cdf_inc_max_global": max(s["cdf_inc_max"] for s in layer_summaries),
        "geom_inv_slope_min_global": min(s["inv_slope_min"] for s in layer_summaries),
        "geom_inv_slope_max_global": max(s["inv_slope_max"] for s in layer_summaries),
    }


@torch.inference_mode()
def inspect_pwl_layer_geometry(flow, Y_probe):
    """
    Y_probe: (B, D) points in [0,1]^D used to probe the conditioning
    Returns per-layer summaries of widths and cdf increments.

    """

    summaries = []

    for layer_idx, layer in enumerate(flow.layers):
        y_A = Y_probe[:, layer.A_dims]
        B = Y_probe.size(0)

        raw_heights = layer.heights_net(y_A).reshape(B, layer.D_B, layer.K - 1)
        raw_widths  = layer.widths_net(y_A).reshape(B, layer.D_B, layer.K)

        heights = layer._heights_from_raw(raw_heights)
        bins = layer._bins_from_raw(raw_widths)

        # widths in x-space
        widths = bins[..., 1:] - bins[..., :-1]   # (B, D_B, K)

        # cdf increments in y-space
        zeros = torch.zeros((B, layer.D_B, 1), device=Y_probe.device, dtype=Y_probe.dtype)
        ones  = torch.ones((B, layer.D_B, 1), device=Y_probe.device, dtype=Y_probe.dtype)
        cdf = torch.cat([zeros, heights, ones], dim=-1)      # (B, D_B, K+1)
        cdf_incs = cdf[..., 1:] - cdf[..., :-1]              # (B, D_B, K)

        inv_slopes = widths / cdf_incs

        summary = {
            "layer": layer_idx,

            "width_min": widths.min().item(),
            "width_mean": widths.mean().item(),
            "width_max": widths.max().item(),

            "cdf_inc_min": cdf_incs.min().item(),
            "cdf_inc_mean": cdf_incs.mean().item(),
            "cdf_inc_max": cdf_incs.max().item(),

            "inv_slope_min": inv_slopes.min().item(),
            "inv_slope_mean": inv_slopes.mean().item(),
            "inv_slope_max": inv_slopes.max().item(),
        }
        summaries.append(summary)

    return summaries


def pwl_geometry_penalty_weighted(
    flow,
    Y_probe,
    layer_slope_lambdas,
    lambda_width=0.0,
    lambda_cdf=0.0,
    eps=1e-12,
):
    """
    Geometry penalty with layer-dependent slope regularisation.
    """
    total_penalty = torch.tensor(0.0, device=Y_probe.device, dtype=Y_probe.dtype)
    layer_penalties = []

    for layer_idx, layer in enumerate(flow.layers):
        y_A = Y_probe[:, layer.A_dims]
        B = Y_probe.size(0)

        raw_heights = layer.heights_net(y_A).reshape(B, layer.D_B, layer.K - 1)
        raw_widths  = layer.widths_net(y_A).reshape(B, layer.D_B, layer.K)

        heights = layer._heights_from_raw(raw_heights)
        bins = layer._bins_from_raw(raw_widths)

        widths = bins[..., 1:] - bins[..., :-1]

        zeros = torch.zeros((B, layer.D_B, 1), device=Y_probe.device, dtype=Y_probe.dtype)
        ones  = torch.ones((B, layer.D_B, 1), device=Y_probe.device, dtype=Y_probe.dtype)
        cdf = torch.cat([zeros, heights, ones], dim=-1)
        cdf_incs = cdf[..., 1:] - cdf[..., :-1]

        log_inv_slope = torch.log(widths + eps) - torch.log(cdf_incs + eps)
        log_width = torch.log(widths + eps)
        log_cdf   = torch.log(cdf_incs + eps)

        slope_pen = (log_inv_slope ** 2).mean()
        width_pen = log_width.var(unbiased=False)
        cdf_pen   = log_cdf.var(unbiased=False)

        lam_slope = layer_slope_lambdas[layer_idx]

        layer_pen = (
            lam_slope * slope_pen
            + lambda_width * width_pen
            + lambda_cdf * cdf_pen
        )

        total_penalty = total_penalty + layer_pen

        layer_penalties.append({
            "layer": layer_idx,
            "lambda_slope": lam_slope,
            "slope_penalty": slope_pen.detach().item(),
            "width_penalty": width_pen.detach().item(),
            "cdf_penalty": cdf_pen.detach().item(),
            "layer_penalty": layer_pen.detach().item(),
        })

    return total_penalty, layer_penalties


def train_loop_decay_diagnostic_2(
    D,
    layer_type,
    A_dims,
    B_dims,
    K,
    hidden_size,
    N,
    epochs,
    lr,
    device,
    dtype,
    ticker=True,
    print_every=20,
    unit_test_every=100,
    unit_test_samples=100_000,
    clip_grad=10.0,
    m_t=173,
    geometry_every=20,
    geometry_probe_size=4096,
):
    torch_dtype = _as_torch_dtype(dtype)

    flow = normalising_flow(
        layer_type=layer_type,
        A_dims=A_dims,
        B_dims=B_dims,
        K=K,
        hidden_size=hidden_size,
        min_bin_width=0.001,
        min_cdf_inc=0.001,
    ).to(device=device, dtype=torch_dtype)

    optimiser = torch.optim.Adam(flow.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.3, patience=20
    )

    # Geometry-regularisation hyperparameters
    layer_slope_lambdas = [9e-4, 2e-4, 1e-4, 1e-4, 1e-4]
    lambda_width = 0.0
    lambda_cdf = 0.0
    penalty_probe_size = 1000

    with torch.no_grad():
        V_scale = torch.rand((5 * N, D), device=device, dtype=torch_dtype)
        X_scale, jac_dets_scale = flow.inverse(V_scale)
        (P, P1, P2, P3), jac_map_scale = element.hypercube_to_momenta(X_scale, m_t=m_t)
        me2_scale = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)
        h_scale = me2_scale * jac_dets_scale * jac_map_scale
        scale = torch.std(h_scale)

    print(f"Initial normalisation scale: {scale.item():.6e}")

    loss_values = torch.zeros(epochs, device=device, dtype=torch_dtype)
    diagnostics = []

    for epoch in range(epochs):
        V = torch.rand((N, D), device=device, dtype=torch_dtype)

        X, jac_dets = flow.inverse(V)
        (P, P1, P2, P3), jac_map = element.hypercube_to_momenta(X, m_t=m_t)
        me2 = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)

        h_evals = me2 * jac_dets * jac_map
        h_norm = h_evals / scale

        loss_main = torch.var(h_norm)

        Y_probe_pen = torch.rand((penalty_probe_size, D), device=device, dtype=torch_dtype)
        geom_penalty, geom_info = pwl_geometry_penalty_weighted(
            flow,
            Y_probe_pen,
            layer_slope_lambdas=layer_slope_lambdas,
            lambda_width=lambda_width,
            lambda_cdf=lambda_cdf,
        )

        loss = loss_main + geom_penalty
        loss_values[epoch] = loss.detach()

        optimiser.zero_grad()
        loss.backward()

        grad_info_preclip = _grad_and_param_norm(flow)

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=clip_grad)

        grad_info_postclip = _grad_and_param_norm(flow)

        optimiser.step()
        #scheduler.step(loss.detach())

        with torch.inference_mode():
            current_lr = optimiser.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "loss": loss.detach().item(),
                "loss_main": loss_main.detach().item(),
                "geom_penalty": geom_penalty.detach().item(),
                "scale": scale.item(),
                "lr": current_lr,

                # store penalty hyperparameters too
                "lambda_width": float(lambda_width),
                "lambda_cdf": float(lambda_cdf),
                "penalty_probe_size": int(penalty_probe_size),
            }

            # store layerwise regularisation info for plotting later
            for info in geom_info:
                li = info["layer"]
                record[f"layer{li}_lambda_slope"] = info["lambda_slope"]
                record[f"layer{li}_slope_penalty"] = info["slope_penalty"]
                record[f"layer{li}_width_penalty"] = info["width_penalty"]
                record[f"layer{li}_cdf_penalty"] = info["cdf_penalty"]
                record[f"layer{li}_layer_penalty"] = info["layer_penalty"]

            record.update(_tensor_stats("jac", jac_dets))
            record.update(_tensor_stats("jac_map", jac_map))
            record.update(_tensor_stats("me2", me2))
            record.update(_tensor_stats("h", h_evals))
            record.update(_tensor_stats("h_norm", h_norm))
            record.update(_tensor_stats("x", X))
            record.update({f"preclip_{k}": v for k, v in grad_info_preclip.items()})
            record.update({f"postclip_{k}": v for k, v in grad_info_postclip.items()})

            # Geometry diagnostics for inspection/plotting
            do_geometry = (geometry_every is not None) and (
                epoch % geometry_every == 0 or epoch == epochs - 1
            )
            if do_geometry:
                Y_probe_geom = torch.rand((geometry_probe_size, D), device=device, dtype=torch_dtype)
                layer_geom = inspect_pwl_layer_geometry(flow, Y_probe_geom)

                record["layer_geometry"] = layer_geom
                record.update(_geometry_summary_across_layers(layer_geom))

                for s in layer_geom:
                    li = s["layer"]
                    record[f"layer{li}_width_min"] = s["width_min"]
                    record[f"layer{li}_width_mean"] = s["width_mean"]
                    record[f"layer{li}_width_max"] = s["width_max"]

                    record[f"layer{li}_cdf_inc_min"] = s["cdf_inc_min"]
                    record[f"layer{li}_cdf_inc_mean"] = s["cdf_inc_mean"]
                    record[f"layer{li}_cdf_inc_max"] = s["cdf_inc_max"]

                    record[f"layer{li}_inv_slope_min"] = s["inv_slope_min"]
                    record[f"layer{li}_inv_slope_mean"] = s["inv_slope_mean"]
                    record[f"layer{li}_inv_slope_max"] = s["inv_slope_max"]

            if unit_test_every is not None and (epoch % unit_test_every == 0 or epoch == epochs - 1):
                record.update(
                    estimate_unit_integral(
                        flow,
                        D,
                        n_test=unit_test_samples,
                        batch_size=min(20_000, unit_test_samples),
                        device=device,
                        dtype=torch_dtype,
                    )
                )

            diagnostics.append(record)

        if ticker and (epoch % print_every == 0 or epoch == epochs - 1):
            msg = (
                f"Epoch {epoch:5d} | "
                f"loss={record['loss']:.6e} | "
                f"main={record['loss_main']:.6e} | "
                f"geom_pen={record['geom_penalty']:.6e} | "
                f"lr={record['lr']:.3e} | "
                f"jac_mean={record['jac_mean']:.6e} | "
                f"jac_std={record['jac_std']:.6e} | "
                f"jac_min={record['jac_min']:.6e} | "
                f"jac_max={record['jac_max']:.6e} | "
                f"jac_map_mean={record['jac_map_mean']:.6e} | "
                f"h_std={record['h_std']:.6e} | "
                f"grad_norm={record['postclip_grad_norm']:.6e}"
            )

            if "unit_int_mean" in record:
                msg += (
                    f" | unit_int={record['unit_int_mean']:.6e}"
                    f" ± {record['unit_int_error']:.2e}"
                )

            if "geom_width_min_global" in record:
                msg += (
                    f" | width[min,max]=({record['geom_width_min_global']:.3e},"
                    f" {record['geom_width_max_global']:.3e})"
                    f" | cdf_inc[min,max]=({record['geom_cdf_inc_min_global']:.3e},"
                    f" {record['geom_cdf_inc_max_global']:.3e})"
                    f" | inv_slope[min,max]=({record['geom_inv_slope_min_global']:.3e},"
                    f" {record['geom_inv_slope_max_global']:.3e})"
                )

            print(msg)

            if (
                record["jac_finite_frac"] < 1.0
                or record["x_finite_frac"] < 1.0
                or record["jac_map_finite_frac"] < 1.0
                or record["me2_finite_frac"] < 1.0
                or record["h_finite_frac"] < 1.0
            ):
                print("  WARNING: non-finite values detected.")

            # print layerwise penalty info
            print("  Layer penalties:")
            for li in range(len(layer_slope_lambdas)):
                print(
                    f"    layer {li}: "
                    f"lambda_slope={record[f'layer{li}_lambda_slope']:.3e} | "
                    f"slope_penalty={record[f'layer{li}_slope_penalty']:.3e} | "
                    f"width_penalty={record[f'layer{li}_width_penalty']:.3e} | "
                    f"cdf_penalty={record[f'layer{li}_cdf_penalty']:.3e} | "
                    f"layer_penalty={record[f'layer{li}_layer_penalty']:.3e}"
                )

            if "layer_geometry" in record:
                print("  Layer geometry:")
                for s in record["layer_geometry"]:
                    print(
                        f"    layer {s['layer']}: "
                        f"width[min,mean,max]=({s['width_min']:.3e}, {s['width_mean']:.3e}, {s['width_max']:.3e}) | "
                        f"cdf_inc[min,mean,max]=({s['cdf_inc_min']:.3e}, {s['cdf_inc_mean']:.3e}, {s['cdf_inc_max']:.3e}) | "
                        f"inv_slope[min,mean,max]=({s['inv_slope_min']:.3e}, {s['inv_slope_mean']:.3e}, {s['inv_slope_max']:.3e})"
                    )

    final_loss = loss.detach()
    return final_loss, loss_values, flow, diagnostics



def distortion_plot(X, h):
  '''
  X: (B, 2) after inverse flow
  h: f(X) * jacobians (function evaluations)
  '''

  fig, ax = plt.subplots(figsize=(8,6))
  plot = ax.scatter(
    X[0].squeeze().cpu(),
    X[1].squeeze().cpu(),
    c=h.cpu(),
    s=6,
    cmap='viridis'
  )
  ax.set_xlabel("$x_1$", fontsize=15)
  ax.set_ylabel("$x_2$", fontsize=15)

  ax.tick_params(axis="both", which="major", direction="out", length=5, labelsize=10)
  ax.tick_params(axis="both", which="minor", direction="out", length=3)
  ax.minorticks_on()

  cbar = plt.colorbar(plot, ax=ax)
  cbar.formatter = ticker.FormatStrFormatter('%.2f')
  cbar.ax.tick_params(labelsize=16)
  cbar.update_ticks()

  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(0.0, 1.0)

  plt.show()


def plot_hist(flow, num_samples, bins, device):
  '''
  Plots histogram of the distribution of x values induced by passing uniformly sampled y values through the flow provided.
  Ideally this should resemble the original function as much as possible.
  '''

  Y = torch.rand(num_samples, 2).to(device)

  X, jacobians = flow.inverse(Y)
  X = X.detach().cpu()

  fig, ax = plt.subplots(figsize=(8,6))

  plot = ax.hist2d(
        X[:,0],
        X[:,1],
        bins=bins,
        range=[[0,1],[0,1]],
        cmap = 'viridis'
  )

  cbar = plt.colorbar(plot[3], ax=ax)
  cbar.ax.tick_params(labelsize=10)
  cbar.update_ticks()

  ax.set_xlabel("$x_1$", fontsize=15)
  ax.set_ylabel("$x_2$", fontsize=15)

  plt.show()


def plot_weights(f, X, jacobians):
  '''
  Computes importance weights using given args and returns a scatter plot.
  '''

  weights = f(X[:,0], X[:,1]) * torch.abs(jacobians)
  weights = weights.detach().cpu()

  fig, ax = plt.subplots(figsize=(8,6))

  sc = ax.scatter(
      X[:,0],
      X[:,1],
      c=weights,
      s=1,
      cmap="viridis"
  )

  cbar = plt.colorbar(sc, ax=ax)
  cbar.ax.tick_params(labelsize=10)
  cbar.update_ticks()

  ax.set_xlim(0.0, 1.0)
  ax.set_ylim(0.0, 1.0)
  plt.show()



