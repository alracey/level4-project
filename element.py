import sys
import torch
import numpy as np


MG_DIR = "/home/alex/MG5_aMC_v3_7_0/tdecay/SubProcesses/P1_t_bepve"
PARAM_CARD = "/home/alex/MG5_aMC_v3_7_0/tdecay/Cards/param_card.dat"

if MG_DIR not in sys.path:
    sys.path.append(MG_DIR)

import matrix2py

matrix2py.py_initialisemodel(PARAM_CARD)

def invert_momenta(p):
        """ fortran/C-python do not order table in the same order"""
        new_p = []
        for i in range(len(p[0])):  new_p.append([0]*len(p))
        for i, onep in enumerate(p):
            for j, x in enumerate(onep):
                new_p[j][i] = x
        return new_p


def get_matrix_element(p, alphas=0.13, nhel=-1):
     '''
     p=[E, px, py, pz]... format (gets inverted)
     '''

     P = invert_momenta(p)
     me2 = matrix2py.py_get_value(P, alphas, nhel)
     #print(me2)
     return me2



def get_p1_vec(E1, costheta1, phi1):
    '''Takes N sets of args.
    Returns 3-momentum for  N p1's.
        Assumes m_1 = 0'''

    sintheta1 = torch.sqrt(1-(costheta1**2))

    px = E1 * sintheta1 * torch.cos(phi1)
    py = E1 * sintheta1 * torch.sin(phi1)
    pz = E1 * costheta1
    tensor = torch.stack([px, py, pz], dim=-1)  #(N, 3)
    return tensor


def get_local_basis(p1_vec):
    '''P1_vec: (N, 3)
    Returns orthonormal basis in p1 to be used for p2'''

    p1_norm = torch.linalg.norm(p1_vec, dim=-1, keepdim=True)
    p1_hat = p1_vec / p1_norm

    ez = torch.tensor([0.0, 0.0, 1.0], device=p1_vec.device, dtype=p1_vec.dtype).expand_as(p1_vec)
    ex = torch.tensor([1.0, 0.0, 0.0], device=p1_vec.device, dtype=p1_vec.dtype).expand_as(p1_vec)

    use_ex = (torch.abs((p1_hat * ez).sum(dim=-1, keepdim=True)) >= 1.0 - 1e-12)
    ref = torch.where(use_ex, ex, ez)

    n1 = torch.linalg.cross(p1_hat, ref, dim=-1)
    n1_hat = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True)       #just in case
    n2 = torch.linalg.cross(p1_hat, n1_hat, dim=-1)
    n2_hat = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True)        #just in case

    return p1_hat, n1_hat, n2_hat       #each have shape ?


def get_costheta2(E1, E2, m_t=173):
    '''Computes N sets of costheta2 for N sets of args'''

    a = m_t**2 -2*m_t*E1 - 2*m_t*E2 + 2*E1*E2
    b = 2*E1*E2
    costheta2 = a / b
    
    return torch.clamp(costheta2, -1.0 + 1e-12, 1.0 - 1e-12)    #avoid issues with sin
    

def get_p2_vec(E2, costheta2, phi2, p1_vec):
    '''Returns 3 momentum for N p2's.  Assumes m_2=0'''

    sintheta2 = torch.sqrt(1-(costheta2**2))

    p1_hat, n1_hat, n2_hat = get_local_basis(p1_vec)
    
    E2 = E2.unsqueeze(-1)
    px = E2 * sintheta2.unsqueeze(-1) * torch.cos(phi2).unsqueeze(-1) * n1_hat
    py = E2 * sintheta2.unsqueeze(-1) * torch.sin(phi2).unsqueeze(-1) * n2_hat
    pz = E2 * costheta2.unsqueeze(-1) * p1_hat
    return px + py + pz


def hypercube_to_momenta(X, m_t=173):
    '''X: (N, 5) on unit interval [0,1]^5
    Computes N sets of valid momenta from the input V using change of variables.
     '''

    u0, u1, u2, u3, u4 = [X[:, i] for i in range(5)]

    #define Y = u0^2 so u0 = sqrt(Y) but u0 from array is just a random number so can take Y = u0

    #mappings
    E1 = 0.5 * m_t * torch.sqrt(u0)
    E2 = (0.5 * m_t - E1) + E1 * u1
    costheta1 = 2 * u2 - 1
    phi1 = 2 * torch.pi * u3
    phi2 = 2 * torch.pi * u4

    jacobian = m_t**2 * (torch.pi)**2 * torch.ones_like(u0)          ##(N)

    costheta2 = get_costheta2(E1, E2, m_t)

    #momenta

    p1_vec = get_p1_vec(E1, costheta1, phi1)
    p2_vec = get_p2_vec(E2, costheta2, phi2, p1_vec)

    P = torch.stack([               #(N, 4)
        torch.full_like(E1, m_t),
        torch.zeros_like(E1),
        torch.zeros_like(E1),
        torch.zeros_like(E1)
    ], dim=-1)
        
    P1 = torch.cat([E1.unsqueeze(-1), p1_vec], dim=-1)      #(N, 4)
    P2 = torch.cat([E2.unsqueeze(-1), p2_vec], dim=-1)      #(N, 4)
    P3 = P - P1 - P2                                        #(N, 4)

    return (P, P1, P2, P3), jacobian


def batch_element_eval(P, P1, P2, P3, device, dtype):

    if dtype is torch.float64:
        np_dtype = np.float64
    elif dtype is torch.float32:
        np_dtype = np.float32
    else:
        np_dtype = dtype

    P_np = P.detach().cpu().numpy()
    P1_np = P1.detach().cpu().numpy()
    P2_np = P2.detach().cpu().numpy()
    P3_np = P3.detach().cpu().numpy()
    N = P_np.shape[0]
    #print(P_np.shape)

    values = np.empty(N, dtype=np_dtype)

    for i in range(N):
        p = [
            P_np[i],
            P1_np[i],
            P2_np[i],
            P3_np[i]
        ]

        values[i] = get_matrix_element(p, alphas=0.13, nhel=-1)

    return torch.from_numpy(values).to(device)

def check_conservation(P1, P2, P3, m_t, tol=1e-6):
    '''
    Checks energy and momentum conservation for a batch of events.
    '''

    # energies
    E1 = P1[:, 0]
    E2 = P2[:, 0]
    E3 = P3[:, 0]

    # momenta
    p1 = P1[:, 1:]
    p2 = P2[:, 1:]
    p3 = P3[:, 1:]

    # checks
    energy_residual = E1 + E2 + E3 - m_t
    momentum_residual = p1 + p2 + p3

    print("Max |E1+E2+E3 - m_t|:", energy_residual.abs().max().item())
    print("Max |p1+p2+p3|:", momentum_residual.abs().max().item())

    # optional assertions
    assert energy_residual.abs().max() < tol, "Energy conservation violated"
    assert momentum_residual.abs().max() < tol, "Momentum conservation violated"

    print("Conservation checks passed")
