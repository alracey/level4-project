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
  def __init__(self, D_A, D_B, A_dims, B_dims, K, hidden_size):

    super().__init__()

    #attributes

    self.D_A = D_A
    self.D_B = D_B
    self.K = K
    self.register_buffer("A_dims", A_dims)
    self.register_buffer("B_dims", B_dims)

    #networks

    self.heights_net = flownet_V(input_size=D_A, hidden_size=hidden_size, output_size=D_B*(K-1))

    self.widths_net = flownet_W(input_size=D_A, hidden_size=hidden_size, output_size=D_B*K)


  def forward(self, x):

    B = x.size(0)

    x_A = x[:, self.A_dims]   #apply mask
    x_B = x[:, self.B_dims]

    raw_heights = self.heights_net(x_A).reshape(B, self.D_B, (self.K - 1))    #evaluate neural networks with x_A
    raw_widths = self.widths_net(x_A).reshape(B, self.D_B, self.K)

    heights = pwl_raw_heights_to_params(raw_heights)
    bins = raw_widths_to_bins(raw_widths)

    y_A, y_B = pwl_g_coupling(x_A, x_B, heights, bins)    #apply the transform to x_B

    #reconstruct the y vector

    y = torch.zeros_like(x)
    y[:, self.A_dims] = y_A
    y[:, self.B_dims] = y_B

    return y


  def inverse(self, y):

    B = y.size(0)

    y_A = y[:, self.A_dims]
    y_B = y[:, self.B_dims]

    raw_heights = self.heights_net(y_A).reshape(B, self.D_B, (self.K - 1))    #evaluate neural networks with x_A
    raw_widths = self.widths_net(y_A).reshape(B, self.D_B, self.K)

    heights = pwl_raw_heights_to_params(raw_heights)
    bins = raw_widths_to_bins(raw_widths)

    x_B, jac_det = pwl_inverse_transform(y_B, self.B_dims, heights, bins)
    x_A = y_A

    x = torch.zeros_like(y)
    x[:, self.A_dims] = x_A
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


def normalising_flow(layer_type, A_dims, B_dims, K, hidden_size):

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

    layer = layer_type(
        D_A = D_A,
        D_B = D_B,
        A_dims = A_dims[i],
        B_dims = B_dims[i],
        K = K,                  #using same K for each layer for now
        hidden_size = hidden_size
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
  
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.3, patience=100)

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

    #normalise

    h_norm = h_evals / scale

    #compute loss
    loss = torch.var(h_norm)
    loss_values[epoch] = loss.detach()

    #scheduler.step(loss.detach())

    optimiser.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

    optimiser.step()

    if ticker and epoch % 20 == 0:
      print(f"Epoch: {epoch} | Loss: {loss:.8e}")
      print("Mean value: ", torch.mean(h_evals).detach().item())
      #print("Std h :", torch.std(h_evals).detach().item())
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

  def __init__(self, D_A, D_B, A_dims, B_dims, K, hidden_size):

    super().__init__()

    #attributes

    self.D_A = D_A
    self.D_B = D_B
    self.K = K
    self.register_buffer("A_dims", A_dims)
    self.register_buffer("B_dims", B_dims)

    #networks

    self.heights_net = flownet_V(input_size=D_A, hidden_size=hidden_size, output_size=(D_B * (K+1)))

    self.widths_net = flownet_W(input_size=D_A, hidden_size=hidden_size, output_size=(D_B * K))

  
  def forward(self, x):

    B = x.size(0)

    x_A = x[:, self.A_dims]   #apply mask
    x_B = x[:, self.B_dims]

    raw_heights = self.heights_net(x_A).reshape(B, self.D_B, (self.K+1))
    raw_heights = torch.nn.functional.softplus(raw_heights) + torch.finfo(x.dtype).eps # Ensure positivity
    raw_widths = self.widths_net(x_A).reshape(B, self.D_B, self.K)

    bins = raw_widths_to_bins(raw_widths)

    y_A, y_B = pwq_g_coupling(x_A, x_B, raw_heights, bins)    #apply transform to x_B

    #reconstruct the y vector

    y = torch.zeros_like(x)
    y[:, self.A_dims] = y_A
    y[:, self.B_dims] = y_B

    return y


  def inverse(self, y):

    B = y.size(0)

    y_A = y[:, self.A_dims]
    y_B = y[:, self.B_dims]

    raw_heights = self.heights_net(y_A).reshape(B, self.D_B, (self.K+1))
    raw_heights = torch.nn.functional.softplus(raw_heights) + torch.finfo(y.dtype).eps # Ensure positivity
    raw_widths = self.widths_net(y_A).reshape(B, self.D_B, self.K)

    bins = raw_widths_to_bins(raw_widths)
    x_B, jac_det = pwq_inverse_transform(y_B, self.B_dims, raw_heights, bins)
    x_A = y_A

    x = torch.zeros_like(y)
    x[:, self.A_dims] = x_A
    x[:, self.B_dims] = x_B

    return x, jac_det