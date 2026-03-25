import vegas
import math
import numpy as np
import torch
import gc

#example from VEGAS documentation tutorial#
def f(x):
    dx2 = 0
    for d in range(4):
        dx2 += (x[d] - 0.5)**2
    return math.exp(-dx2 * 100) * 1013.2118364296088

integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])

result = integ(f, nitn=10, neval=1000)

print(result.summary())


#top decay width estimating

import element

def vegas_decay_width_estimator(
        B_total,
        nitn_adapt=10,
        neval_adapt=20_000,
        nitn_prod=10,
        neval_prod=100_000,
        nhcube_batch=1_000      #analogous to chunk_size in flow estimator
):
    '''
    m_t assumed to be 173 GeV
    Returns:
        mean         : vegas decay-width estimate
        stderr       : vegas integration error
        events_all   : tensor of shape (N, 4, 4)
        weights_all  : tensor of shape (N,) with vegas event weights
        integ        : adapted vegas integrator (useful for inspection / reuse)
        result       : raw vegas result object
    '''
    m_t=173

    prefactor = 1 / ((2 * torch.pi)**5 * 16 *m_t)
    dtype = torch.float64
    device='cpu'

    #optional conservation check as always
    X_test = torch.rand((1000, 5), device=device, dtype=dtype)
    (_, P1t, P2t, P3t), _ = element.hypercube_to_momenta(X_test, m_t)
    element.check_conservation(P1t, P2t, P3t, m_t, tol=1e-10)
    del X_test, P1t, P2t, P3t

    @vegas.lbatchintegrand      #batched operations for memory
    def f(x_numpy):

        '''
        x_numpy: array (B, 5) which vegas will pass
        Returns: array (B,)
        '''

        #numpy to pytorch
        X = torch.from_numpy(np.ascontiguousarray(x_numpy)).to(device=device, dtype=dtype)

        (P, P1, P2, P3), jac_map = element.hypercube_to_momenta(X, m_t)
        me2_vals = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)

        vals = prefactor * me2_vals * jac_map
        return vals.detach().cpu().numpy()
    

    #vegas integrator which uses unit hypercube for D=5
    integ = vegas.Integrator(5 * [[0.0, 1.0]], nhcube_batch=nhcube_batch)

    #iterative adaptation phase (only fair)
    integ(f, nitn=nitn_adapt, neval=neval_adapt)

    #non-adaptive accurate integral estimation
    result = integ(f, nitn=nitn_prod, neval=neval_prod, adapt=False)

    mean = torch.tensor(result.mean, dtype=dtype)
    stderr = torch.tensor(result.sdev, dtype=dtype)

    #we can also generate weighted events from the adapted vegas integrator
    
    events_all = []
    weights_all = []
    n_total = 0
    
    while n_total < B_total:
        for x_numpy, wgt_np in integ.random_batch():
            X = torch.from_numpy(np.ascontiguousarray(x_numpy)).to(device=device, dtype=dtype)
            wgt = torch.from_numpy(np.ascontiguousarray(wgt_np)).to(device=device, dtype=dtype)

            (P, P1, P2, P3), jac_map = element.hypercube_to_momenta(X, m_t)
            me2_vals = element.batch_element_eval(P, P1, P2, P3, device=device, dtype=dtype)

            evals = prefactor * me2_vals * jac_map

            #vegas event weight
            weights = wgt * evals
            
            events = torch.stack((P, P1, P2, P3), dim=1)    #(B, 4, 4)

            # Trim final batch if we overshoot B_total
            remaining = B_total - n_total
            if events.size(0) > remaining:
                events = events[:remaining]
                weights = weights[:remaining]

            events_all.append(events.detach().clone())
            weights_all.append(weights.detach().clone())

            n_total += events.size(0)

            del X, wgt, P, P1, P2, P3, jac_map, me2_vals, evals, weights, events

            if n_total >= B_total:
                break

        gc.collect()

    events_all = torch.cat(events_all, dim=0)   #(B_total, 4, 4)
    weights_all = torch.cat(weights_all, dim=0) #(B_total,)

    return mean, stderr, events_all, weights_all, integ, result
