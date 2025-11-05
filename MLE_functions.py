import jax
from jax import grad,jacfwd,jit
import jax.numpy as jnp
from jax.scipy import linalg
from functools import partial
import jax.lax as lax

from jax import config
config.update("jax_enable_x64", True)

import numpy as np
from scipy import signal, optimize
from typing import List, Sequence, Tuple, Union




# ----------------------------------------------------------------------
#  Non Linear Least-square for initialization
# ----------------------------------------------------------------------


def _count_params(nb_orders: Sequence[int], na_orders: Sequence[int]) -> int:
    """Total number of free parameters across all inputs."""
    return sum(nb + 1 + na for nb, na in zip(nb_orders, na_orders))


def _unpack_theta(theta: np.ndarray,
                  nb_orders: Sequence[int],
                  na_orders: Sequence[int]
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Decompose flat vector into per-input numerator and denominator arrays."""
    bs, as_ = [], []
    k = 0
    for nb, na in zip(nb_orders, na_orders):
        b = theta[k: k + nb + 1]
        k += nb + 1
        a = np.concatenate(([1.0], theta[k: k + na]))
        k += na
        bs.append(b)
        as_.append(a)
    return bs, as_


def _shift_signal(u: np.ndarray, d: int) -> np.ndarray:
    """Right‑shift signal by d samples (insert zeros at start)."""
    if d == 0:
        return u
    if d >= len(u):
        raise ValueError("Delay {} ≥ signal length {}".format(d, len(u)))
    return np.concatenate([np.zeros(d, dtype=u.dtype), u[:-d]])


def _simulate_miso(u: np.ndarray,
                   theta: np.ndarray,
                   nb_orders: Sequence[int],
                   na_orders: Sequence[int],
                   delays: Sequence[int]) -> np.ndarray:
    bs, as_ = _unpack_theta(theta, nb_orders, na_orders)
    y_hat = np.zeros_like(u[0], dtype=float)
    for ui, di, bi, ai in zip(u, delays, bs, as_):
        ui_delayed = _shift_signal(ui, di)
        y_hat += signal.lfilter(bi, ai, ui_delayed)
    return y_hat


def _residual(theta: np.ndarray,
              u: np.ndarray,
              y: np.ndarray,
              nb_orders: Sequence[int],
              na_orders: Sequence[int],
              delays: Sequence[int]) -> np.ndarray:
    return _simulate_miso(u, theta, nb_orders, na_orders, delays) - y


def estimate_miso(u: Sequence[np.ndarray],
                  y: np.ndarray,
                  nb_orders: Sequence[int],
                  na_orders: Sequence[int],
                  delays: Union[int, Sequence[int]] = 0,
                  theta0: np.ndarray = None,
                  **ls_kwargs):
    """Identify a delayed MISO system via non‑linear least squares.

    Parameters
    ----------
    u         : list/array (m, N)   Input signals.
    y         : array (N,)          Output.
    nb_orders : list[int]           Numerator orders per input.
    na_orders : list[int]           Denominator orders per input.
    delays    : int or list[int]    Fixed sample delay per input.
    theta0    : optional initial parameter vector.
    ls_kwargs : extra kwargs for scipy.optimize.least_squares.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u[None, :]
    m, N = u.shape
    assert len(nb_orders) == len(na_orders) == m, "orders length mismatch"

    if isinstance(delays, int):
        delays = [delays] * m
    delays = list(delays)
    assert len(delays) == m, "delays length mismatch"
    if any(d < 0 for d in delays):
        raise ValueError("Delays must be non‑negative integers")

    n_par = _count_params(nb_orders, na_orders)
    if theta0 is None:
        theta0 = np.zeros(n_par)
        theta0[0::(max(nb_orders) + 1)] = 0.1  # gentle initialisation

    ls_defaults = dict(method="trf", verbose=2)
    ls_defaults.update(ls_kwargs)

    res = optimize.least_squares(_residual, theta0,
                                 args=(u, y, nb_orders, na_orders, delays),
                                 **ls_defaults)

    bs, as_ = _unpack_theta(res.x, nb_orders, na_orders)
    return bs, as_, res


# ----------------------------------------------------------------------
#  Code for MLE
# ----------------------------------------------------------------------


def permutation_mat(p):
    size_P = len(p)
    temp = jnp.eye(size_P)
    P = temp[:,p[0]-1].reshape(-1,1)
    for index in p[1:]:
        P = jnp.block([P,temp[:,index-1].reshape(-1,1)])
    return P


# ----------------------------------------------------------------------
#  Previous code - not JIT compatible, but easier to understand
# ----------------------------------------------------------------------

""" 
def gen_Ty(a,N):
    n_zeros = N-len(a)-1
    col = jnp.append(1,a)
    col = jnp.append(col,jnp.linspace(0,0,n_zeros))
    Ty = linalg.toeplitz(col,r=jnp.zeros(N))
    return Ty


def gen_Tu(b,N,dif_ab):
    n_zeros = N-len(b)-dif_ab
    col = jnp.append(jnp.linspace(0,0,dif_ab),b)
    col = jnp.append(col,jnp.linspace(0,0,n_zeros))
    Tu = linalg.toeplitz(col,r=jnp.zeros(N))
    return Tu


def gen_Ty_c(a,c,N):
    Ta = gen_Ty(a,N)
    Tc = gen_Ty(c,N)
    return jnp.linalg.inv(Tc)@Ta


def gen_Tu_c(b,c,N,dif_ab):
    Tb =  gen_Tu(b,N,dif_ab)
    Tc = gen_Ty(c,N)
    return jnp.linalg.inv(Tc)@Tb


def gen_A1(theta,n_ab,Permut,N):
    pos = 0
    if n_ab.shape[0] == 2:
        for i in range(len(n_ab[0])):
            na = n_ab[0][i]
            nb = n_ab[1][i]
            if pos==0:
                blk1 = gen_Ty(theta[pos:pos+na],N)
                blk2 = gen_Tu(theta[pos+na:pos+na+nb],N,na+1-nb)
            else:
                blk1 = linalg.block_diag(blk1,gen_Ty(theta[pos:pos+na],N))
                blk2 = linalg.block_diag(blk2,gen_Tu(theta[pos+na:pos+na+nb],N,na+1-nb))
            pos += (na+nb)


    if n_ab.shape[0] == 3:
          for i in range(len(n_ab[0])):
            na = n_ab[0][i]
            nb = n_ab[1][i]
            nc = n_ab[2][i]
            if pos==0:
                blk1 = gen_Ty_c(theta[pos:pos+na],theta[pos+na+nb:pos+na+nb+nc],N)
                blk2 = gen_Tu_c(theta[pos+na:pos+na+nb],theta[pos+na+nb:pos+na+nb+nc],N,na+1-nb)
            else:
                blk1 = linalg.block_diag(blk1,gen_Ty_c(theta[pos:pos+na],theta[pos+na+nb:pos+na+nb+nc],N))
                blk2 = linalg.block_diag(blk2,gen_Tu_c(theta[pos+na:pos+na+nb],theta[pos+na+nb:pos+na+nb+nc],N,na+1-nb))
            pos += (na+nb+nc)      

    return jnp.block([blk1,-blk2])@jnp.kron(Permut,jnp.eye(N))
"""


# ----------------------------------------------------------------------
#  JIT-compatible implementation of the functions above
# ----------------------------------------------------------------------


# ---------------------------------------------------------------------
#  Toeplitz helpers
# ---------------------------------------------------------------------
def _toeplitz_firstcol(col, N):
    """
    Lower-triangular Toeplitz whose first column is `col`
    and whose first row is all zeros (shape = (N, N)).
    """
    # jsp.toeplitz is available in recent JAX builds
    return linalg.toeplitz(col, r=jnp.zeros(N, col.dtype))


# ---------------------------------------------------------------------
#  Tᵧ  (output) --------------------------------------------------------
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=("N",))
def gen_Ty(a: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    Toeplitz with first column [1, a₀, a₁, …, a_{na-1}, 0, …, 0]ᵀ.
    Valid when len(a) + 1 ≤ N.
    """
    n_zeros = N - a.shape[0] - 1                            # static
    col = jnp.concatenate(
        [jnp.ones((1,), a.dtype),              # leading 1
         a,
         jnp.zeros((n_zeros,), a.dtype)]
    )
    return _toeplitz_firstcol(col, N)


# ---------------------------------------------------------------------
#  Tᵤ  (input) ---------------------------------------------------------
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=("N", "dif_ab"))
def gen_Tu(b: jnp.ndarray, N: int, dif_ab: int) -> jnp.ndarray:
    """
    Toeplitz with first column
    [0…0︸dif_ab, b₀, …, b_{nb-1}, 0, …, 0]ᵀ.
    """
    n_zeros = N - b.shape[0] - dif_ab                       # static
    col = jnp.concatenate(
        [jnp.zeros((dif_ab,), b.dtype),
         b,
         jnp.zeros((n_zeros,), b.dtype)]
    )
    return _toeplitz_firstcol(col, N)


# ---------------------------------------------------------------------
#  Colour-corrected variants  Tᵧᶜ , Tᵤᶜ ------------------------------
# ---------------------------------------------------------------------
@partial(jax.jit, static_argnames=("N",))
def gen_Ty_c(a: jnp.ndarray, c: jnp.ndarray, N: int) -> jnp.ndarray:
    """
    (Tᶜ)⁻¹ Tᵧ   — solved with `linalg.solve` instead of an explicit inverse.
    """
    Ta = gen_Ty(a, N)
    Tc = gen_Ty(c, N)
    return jnp.linalg.solve(Tc, Ta)          # Tc⁻¹ @ Ta


@partial(jax.jit, static_argnames=("N", "dif_ab"))
def gen_Tu_c(b: jnp.ndarray,
             c: jnp.ndarray,
             N: int,
             dif_ab: int) -> jnp.ndarray:
    """
    (Tᶜ)⁻¹ Tᵤ   — colour-corrected input Toeplitz.
    """
    Tb = gen_Tu(b, N, dif_ab)
    Tc = gen_Ty(c, N)
    return jnp.linalg.solve(Tc, Tb)          # Tc⁻¹ @ Tb



# ---------------------------------------------------------------------
#  small helper: 1-D dynamic slice with static length
# ---------------------------------------------------------------------

def _slice1d(vec, start_idx: int, length: int):
    """Return vec[start_idx : start_idx + length] with dynamic_slice."""
    return lax.dynamic_slice(vec, (start_idx,), (length,))

# ---------------------------------------------------------------------
#  top-level routine ---------------------------------------------------
# ---------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_ab", "N"))
def gen_A1(theta: jnp.ndarray,
           n_ab: tuple,     # static (tuple of tuples)
           Permut: jnp.ndarray,
           N: int) -> jnp.ndarray:
    """
    Parameters
    ----------
    theta  : 1-D array of parameters
    n_ab   : tuple whose length is 2 (na, nb) or 3 (na, nb, nc); each item is
             itself a tuple of ints, one element per channel
             Example:  ((3,2,4), (2,1,1))  # three channels, na=[3,2,4], nb=[2,1,1]
    Permut : (m,m) permutation matrix
    N      : horizon / number of rows in each Toeplitz block

    Returns
    -------
    A1     : The assembled block matrix, shape (m*N, m*N)
    """

    blk1_list, blk2_list = [], []
    pos = 0  # purely Python, so compile-time constant inside the trace

    if len(n_ab) == 2:
        na_vec, nb_vec = n_ab
        for na, nb in zip(na_vec, nb_vec):                  # static loop
            theta_y = _slice1d(theta, pos,         na)
            theta_u = _slice1d(theta, pos + na,    nb)

            blk1_list.append(gen_Ty(theta_y, N))
            blk2_list.append(gen_Tu(theta_u, N, na + 1 - nb))

            pos += na + nb

    elif len(n_ab) == 3:
        na_vec, nb_vec, nc_vec = n_ab
        for na, nb, nc in zip(na_vec, nb_vec, nc_vec):      # static loop
            theta_y = _slice1d(theta, pos,              na)
            theta_u = _slice1d(theta, pos + na,         nb)
            theta_c = _slice1d(theta, pos + na + nb,    nc)

            blk1_list.append(gen_Ty_c(theta_y, theta_c, N))
            blk2_list.append(gen_Tu_c(theta_u, theta_c, N, na + 1 - nb))

            pos += na + nb + nc
    else:
        raise ValueError("`n_ab` must have 2 or 3 rows (na/nb[/nc]).")

    blk1 = linalg.block_diag(*blk1_list)
    blk2 = linalg.block_diag(*blk2_list)

    A1 = jnp.block([[blk1, -blk2]])
    return A1 @ jnp.kron(Permut, jnp.eye(N))




def gen_A2B2(Lamb,Delta,Permut,M,N,r):
    A2 = jnp.block([-jnp.kron(Lamb,jnp.eye(N)), jnp.eye(M*N)])@jnp.kron(Permut,jnp.eye(N))
    B2 = -jnp.kron(Delta,jnp.eye(N))@r
    return A2,B2


def get_transform_matrices(A2o,A2m,B2):
    m2,no = A2o.shape
    _,nm = A2m.shape
    U,S,V = jnp.linalg.svd(A2m,full_matrices=True)
    V = V.T
    m21 = len(S)
    nm1 = m21
    m22 = m2-m21
    nm2 = nm - nm1
    U1 = U[:,:m21]
    U2 = U[:,m21:]
    V1 = V[:,:nm1]
    V2 = V[:,nm1:]
    
    Sigma1 = jnp.diag(S)
    barA2o2 = U2.T@A2o
    
    W,R = jnp.linalg.qr(barA2o2.T,mode="complete")
    try:
        nW1 = jnp.linalg.matrix_rank(R)
    except:
        nW1 = 0
    nW2 = no - nW1
    
    W1 = W[:,:nW1]
    W2 = W[:,nW1:]
    
    hatA2m = V1@jnp.linalg.pinv(Sigma1)@U1.T
    
    Y1 = W1@jnp.linalg.pinv(U2.T@A2o@W1)@U2.T
    
    To_phi = jnp.block([[jnp.eye(no)],[-hatA2m@A2o]])@W2
    Tm_phi = jnp.block([[jnp.zeros((no,nm2))],[V2]])
    T_gamma = jnp.block([[-Y1@B2], [hatA2m@(A2o@Y1 - jnp.eye(m2))@B2]]);
    
    return To_phi,Tm_phi,T_gamma,W2,V2

    

def eval_cost_func(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo):

    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])
        
    bar_xo2 = W2.T@xo
    m = bar_xo2.shape[0]
    

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    L_PhiSquared = linalg.cholesky(Phi.T@Phi)
    ln_det_PhiSquared = 2*(jnp.log(jnp.diag(L_PhiSquared))).sum()


    Z = Phi_m.T@Phi_m
    #P = np.eye(Phi_m.shape[0]) - Phi_m@jnp.linalg.inv(Z)@Phi_m.T                                                   #previous code with inverse computation
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    
    
    L_Z = linalg.cholesky(Z)
    ln_det_Z = 2*(jnp.log(jnp.diag(L_Z))).sum()
    #return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@P@Phi_xo_Gamma)+(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)      #previous code with inverse computation
    return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma)-(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z+m/2)

    
grad_theta = grad(eval_cost_func, argnums=0)
Hessian_theta = jacfwd(lambda theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo : grad_theta(theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo),argnums=0)



def get_same_lambda(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo):

    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma
        
    bar_xo2 = W2.T@xo
    m = bar_xo2.shape[0]
    
    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma

    Z = Phi_m.T@Phi_m
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    
    return jnp.sum((1/m)*PxPhi_xo_Gamma.T@PxPhi_xo_Gamma)
    
    
def get_ab(theta,n_ab):
    a = []
    b = []
    pos = 0

    if n_ab.shape[0] == 2:
        for i in range(len(n_ab[0])):
            na = n_ab[0][i]
            nb = n_ab[1][i]
            a.append(theta[pos:pos+na])
            b.append(theta[pos+na:pos+na+nb])
            pos += (na+nb)
        return a,b

    if n_ab.shape[0] == 3:
        c = []
        for i in range(len(n_ab[0])):
            na = n_ab[0][i]
            nb = n_ab[1][i]
            nc = n_ab[2][i]
            a.append(theta[pos:pos+na])
            b.append(theta[pos+na:pos+na+nb])
            c.append(theta[pos+na+nb:pos+na+nb+nc])
            pos += (na+nb+nc)
        return a,b,c




def get_xoxm(states,list_xo,N,M):
    p = jnp.arange(2*M)+1
    p = np.concatenate([list_xo,p])
    p_new = []
    for i in p:
        if not i in p_new:
            p_new.append(i)
    p = p_new
    Permut = permutation_mat(p)
    sorted_states = jnp.kron(Permut,jnp.eye(N)).T@states
    xo = sorted_states[:len(list_xo)*N]
    xm = sorted_states[len(list_xo)*N:]
    return xo,xm,Permut




""" 
def eval_cost_func_dif_lambda(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo):

    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    bar_xo2 = W2.T@xo
    m = bar_xo2.shape[0]

    
    pos_lamb = np.sum(n_ab)

    Sig_e_half_inv = (10**theta[pos_lamb])*jnp.eye(N)
    for i in range(1,M):
        Sig_e_half_inv = linalg.block_diag(Sig_e_half_inv, (10**theta[pos_lamb+i])*jnp.eye(N))

    barPhi_m = Sig_e_half_inv@Phi_m

    barPhi = Sig_e_half_inv@jnp.block([Phi_o,Phi_m])

        

    Phi_xo_Gamma = Sig_e_half_inv@(Phi_o@bar_xo2+Gamma)
    L_PhiSquared = linalg.cholesky(barPhi.T@barPhi)
    ln_det_PhiSquared = 2*(jnp.log(jnp.diag(L_PhiSquared))).sum()


    Z = barPhi_m.T@barPhi_m
    #P = np.eye(Phi_m.shape[0]) - Phi_m@jnp.linalg.inv(Z)@Phi_m.T                                                   #previous code with inverse computation
    PxPhi_xo_Gamma = Phi_xo_Gamma - barPhi_m@jnp.linalg.solve(Z,barPhi_m.T@Phi_xo_Gamma)
    
    
    L_Z = linalg.cholesky(Z)
    ln_det_Z = 2*(jnp.log(jnp.diag(L_Z))).sum()
    return jnp.sum((1/2)*(Phi_xo_Gamma.T@PxPhi_xo_Gamma)-(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)

    
grad_theta_dif_lambda = grad(eval_cost_func_dif_lambda, argnums=0)
Hessian_theta_dif_lambda = jacfwd(lambda theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo : grad_theta_dif_lambda(theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo),argnums=0)

"""

# ----------------------------------------------------------------------
#  Better implementation of the function above
# ----------------------------------------------------------------------


def eval_cost_func_dif_lambda(theta,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo):
    
    # ------------- 1. build model matrices -------------
    A1 = gen_A1(theta,n_ab,Permut,N)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    bar_xo2 = W2.T@xo


    # ------------- 2. row scaling by Σ_e^{-½} ----------
    # λ_i^{-½} = 10^{θ_i},  scale rows by λ_i^{-½}
    pos_lamb = np.sum(n_ab)

    lamb = 10.0 ** theta[pos_lamb : pos_lamb + M]
    row_scale = jnp.repeat(lamb, N) # row_scale = jnp.repeat(lamb ** (-0.5), N)

    def rscale(mat):           # fast row scaling
        return mat * row_scale[:, None]

    Phi_o_s = rscale(Phi_o)
    Phi_m_s = rscale(Phi_m)
    Gamma_s = rscale(Gamma)
        
    # ------------- 3. key intermediate vector ----------
    Phi_xo_Gamma = Phi_o_s@bar_xo2+Gamma_s

    # ------------- 4. Gram matrices ---------------------

    G_mm = Phi_m_s.T @ Phi_m_s                            # (n_m × n_m)
    G_oo = Phi_o_s.T @ Phi_o_s                            # (n_o × n_o)
    G_om  = Phi_o_s.T @ Phi_m_s                           # (n_o × n_m)
    


    L_Z  = linalg.cholesky(G_mm, lower=True)                 # Z = Φ_mᵀΣ^{-1}Φ_m
    ln_det_Z = 2.0 * jnp.log(jnp.diag(L_Z)).sum()


    # ---------- 5. Schur complement for log det G ----------
    #   S = G_oo - G_om G_mm^{-1} G_omᵀ,  (n_o × n_o)
    tmp       = linalg.solve_triangular(L_Z, G_om.T, lower=True)        # L_Z^{-1} G_om
    S         = G_oo - (tmp.T @ tmp)                                # uses symmetry
    L_S       = linalg.cholesky(S, lower=True)
    ln_det_S  = 2.0 * jnp.log(jnp.diag(L_S)).sum()

    ln_det_G  = ln_det_Z + ln_det_S


    # ------------- 5. projection term  ------------------
    # P Σ^{-½}(Φ_o x_o + Γ)  without forming P
    tmp = linalg.solve(G_mm, Phi_m_s.T @ Phi_xo_Gamma, assume_a="pos")  # (n_m,)
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m_s @ tmp  

    quad = 0.5 * Phi_xo_Gamma.T@PxPhi_xo_Gamma

    return jnp.sum(quad - 0.5 * ln_det_G + 0.5 * ln_det_Z)

    
grad_theta_dif_lambda = grad(eval_cost_func_dif_lambda, argnums=0)
Hessian_theta_dif_lambda = jacfwd(lambda theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo : grad_theta_dif_lambda(theta,n_ab, Permut, N, M, To_phi, Tm_phi, T_gamma, W2, xo),argnums=0)


def get_different_lambdas(theta,M):
    return (1/(10**theta[-M:]))**(2) # λ_i^{-½} = 10^{θ_i}, i = -M:end



################################################################################################################################################
################################################################################################################################################
################################################################################################################################################

# ----------------------------------------------------------------------
#  Functions CMLE - FEEDBACK A -> C
# ----------------------------------------------------------------------

################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


def gen_A2B2_feedback_AC(Upsilon_A, Delta, r, Upsilon_CA, Permut, N, Upsilon_AC_y_C, N_A,Upsilon_CA_y_A):

    U, s, Vt = np.linalg.svd(jnp.kron(Upsilon_CA,jnp.eye(N)))
    rank_CA = (s > 0.000001).sum()                 
    V = Vt.T
    V1, V2 = V[:, :rank_CA], V[:, rank_CA:]         # V2 spans N(Upsilon_CA)

    U = np.hstack([V2, V1])             # columns reordered so zeros come first

    N_F = N_A - rank_CA
    N_H = N_A + rank_CA

    V = np.block([[U,np.zeros((N_A,N_A))], [jnp.kron(Upsilon_A,jnp.eye(N))@U,np.eye(N_A)]])

    
    Upsilon_CA2_bar = jnp.kron(Upsilon_CA,jnp.eye(N))@U
    Upsilon_CA2_bar = Upsilon_CA2_bar[:,N_F:]

    A2 = np.block([np.zeros((N_H,N_F)), np.eye(N_H)])@np.linalg.inv(V)@jnp.kron(Permut,jnp.eye(N))
    H = np.block([[Upsilon_CA2_bar.T@Upsilon_CA2_bar, np.zeros((rank_CA,N_A))], [np.zeros((N_A,rank_CA)),np.eye(N_A)]])
    B2 = np.linalg.inv(H)@np.block([[-Upsilon_CA2_bar.T@Upsilon_CA_y_A], [-jnp.kron(Delta,np.eye(N))@r-Upsilon_AC_y_C]])


    return A2,B2,U,V,N_F,N_H



@partial(jax.jit, static_argnames=("n_ab", "N", "N_F", "N_H"))
def gen_A1_feedback_AC(theta: jnp.ndarray,
                       n_ab: tuple,             
                       Permut: jnp.ndarray,
                       Upsilon_A: jnp.ndarray,
                       U: jnp.ndarray,
                       V: jnp.ndarray,
                       N: int,
                       N_F: int,
                       N_H: int) -> jnp.ndarray:
    """
    JIT-friendly. Supports n_ab with 2 rows (na, nb) or 3 rows (na, nb, nc).
    Uses reverse Cholesky: F = L.T @ L, where L = chol(F).T
    """

    blk1_list, blk2_list = [], []
    pos = 0

    if len(n_ab) == 2:
        na_vec, nb_vec = n_ab
        for na, nb in zip(na_vec, nb_vec):   # static loop
            theta_y = theta[pos:pos + na]
            theta_u = theta[pos + na:pos + na + nb]

            blk1_list.append(gen_Ty(theta_y, N))
            blk2_list.append(gen_Tu(theta_u, N, na + 1 - nb))

            pos += (na + nb)

    else:  # len(n_ab) == 3
        na_vec, nb_vec, nc_vec = n_ab
        for na, nb, nc in zip(na_vec, nb_vec, nc_vec):  # static loop
            theta_y = theta[pos:pos + na]
            theta_u = theta[pos + na:pos + na + nb]
            theta_c = theta[pos + na + nb:pos + na + nb + nc]

            # Coupled versions when c-terms are present
            blk1_list.append(gen_Ty_c(theta_y, theta_c, N))
            blk2_list.append(gen_Tu_c(theta_u, theta_c, N, na + 1 - nb))

            pos += (na + nb + nc)

    blk1 = linalg.block_diag(*blk1_list)
    blk2 = linalg.block_diag(*blk2_list)

    # ----- core algebra -----
    kron_A = jnp.kron(Upsilon_A, jnp.eye(N))
    factor = blk1 @ U - blk2 @ kron_A @ U

    F_full = factor.T @ factor

    # G_full = [F_full[:, N_F:],  -(factor.T @ blk2)]
    G_full = jnp.concatenate([F_full[:, N_F:], -(factor.T @ blk2)], axis=1)
    G = G_full[:N_F, :N_H]

    # Trim F and compute reverse Cholesky: F = L.T @ L
    F = F_full[:N_F, :N_F]
    L_lower = jnp.linalg.cholesky(F)   # F = L_lower @ L_lower.T
    L = L_lower.T                      # so F = L.T @ L

    # Solve F X = G via Cholesky (avoid inv)
    Y = linalg.solve_triangular(L_lower, G, lower=True)
    X = linalg.solve_triangular(L_lower.T, Y, lower=False)

    # Assemble [L, L @ X]
    LL = jnp.concatenate([L, L @ X], axis=1)

    # Right-multiply by inv(V) stably: LL @ inv(V) = solve(V.T, LL.T).T
    LVinv = linalg.solve(V.T, LL.T).T

    A1 = LVinv @ jnp.kron(Permut, jnp.eye(N))
    return A1





def eval_cost_func_feedback_AC(theta,n_ab,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo):
    
    
    A1 = gen_A1_feedback_AC(theta,n_ab,Permut,Upsilon_A,U,V,N,N_F,N_H)

    Phi_o = A1@To_phi
    Phi_m = A1@Tm_phi
    Gamma = A1@T_gamma

    Phi = jnp.block([Phi_o,Phi_m])
        
    bar_xo2 = W2.T@xo
    m = bar_xo2.shape[0]
    

    Phi_xo_Gamma = Phi_o@bar_xo2+Gamma
    L_PhiSquared = linalg.cholesky(Phi.T@Phi)
    ln_det_PhiSquared = 2*(jnp.log(jnp.diag(L_PhiSquared))).sum()


    Z = Phi_m.T@Phi_m
    #P = np.eye(Phi_m.shape[0]) - Phi_m@jnp.linalg.inv(Z)@Phi_m.T                                                   #previous code with inverse computation
    PxPhi_xo_Gamma = Phi_xo_Gamma - Phi_m@jnp.linalg.solve(Z,Phi_m.T@Phi_xo_Gamma)
    
    
    L_Z = linalg.cholesky(Z)
    ln_det_Z = 2*(jnp.log(jnp.diag(L_Z))).sum()
    #return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@P@Phi_xo_Gamma)+(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)      #previous code with inverse computation
    return jnp.sum((m/2)*jnp.log((1/m)*Phi_xo_Gamma.T@PxPhi_xo_Gamma)-(1/2)*ln_det_PhiSquared+(1/2)*ln_det_Z)


grad_theta_feedback_AC = grad(eval_cost_func_feedback_AC, argnums=0)
Hessian_theta_feedback_AC = jacfwd(lambda theta,n_ab,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo : grad_theta_feedback_AC(theta,n_ab,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo),argnums=0)
