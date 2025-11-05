from MLE_functions import *
from scipy.io import loadmat
from scipy.optimize import minimize
import time


from jax import config
config.update("jax_enable_x64", True)

tolerance = 1e-5

########## System parameters

n_ab = ((2,2,2),(2,2,2),(2,2,2)) #n_ab = [[na1,na2,na3],[nb1,nb2,nb3]]

M = 3 #number of systems
N = 500 #number of samples;
obs = [1,2,3]
n_obs = len(obs)

N_A = M*N
N_o = n_obs*N

#Interconections
Upsilon_A = jnp.array([[0,0,0],[1,0,1],[0,1,0]])
Delta = jnp.array([[0,0],[1,0],[0,1]])


Upsilon_CA = jnp.array([[0,0,0],[0,0,1]])


########## Loading data:

data = loadmat('data/data_estimation.mat')


y6 = data["x"][2500:3000]
y3 = data["x"][1000:1500]
r = jnp.block([data["r"][:,0].reshape(-1,1), data["r"][:,1].reshape(-1,1)])
x = jnp.concatenate([data["x"][0:1500], data["x"][3500:5000]]).reshape(-1,1)


Upsilon_AC_y_C = np.block([[y6],[np.zeros((N,1))],[np.zeros((N,1))]])
Upsilon_CA_y_A = np.block([[np.zeros((N,1))],[y3]])


## Initial theta
theta_init = 0.1*jnp.ones(12)
c_init = jnp.array([0.1, 0.1])
n_ab_arx = ((2,2,2),(2,2,2))




r_sig = jnp.concatenate([r[:,0],r[:,1]]).reshape(-1,1)

xo,_,Permut = get_xoxm(x,obs,N,M)

A2,B2,U,V,N_F,N_H = gen_A2B2_feedback_AC(Upsilon_A, Delta, r_sig, Upsilon_CA, Permut, N, Upsilon_AC_y_C, N_A,Upsilon_CA_y_A)
A2o = A2[:,:N_o]
A2m = A2[:,N_o:]

To_phi, Tm_phi, T_gamma, W2, V2 = get_transform_matrices(A2o,A2m,B2)




@jit
def f_CMLE_arx(theta):
    f = eval_cost_func_feedback_AC(theta.reshape(-1),n_ab_arx,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo)
    f = jnp.where(jnp.isnan(f), 1.0e13, f)
    return f


@jit
def g_CMLE_arx(theta):
    return grad_theta_feedback_AC(theta.reshape(-1),n_ab_arx,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo).reshape(-1)


@jit
def H_CMLE_arx(theta):
    return Hessian_theta_feedback_AC(theta.reshape(-1),n_ab_arx,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])


@jit
def f_CMLE(theta):
    f = eval_cost_func_feedback_AC(theta.reshape(-1),n_ab,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo)
    f = jnp.where(jnp.isnan(f), 1.0e13, f)
    return f


@jit
def g_CMLE(theta):
    return grad_theta_feedback_AC(theta.reshape(-1),n_ab,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo).reshape(-1)


@jit
def H_CMLE(theta):
    return Hessian_theta_feedback_AC(theta.reshape(-1),n_ab,Permut,Upsilon_A,U,V,N,N_F,N_H,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])




startTime = time.time()

# ----------------------------------------------------------------------
#  CMLE of ARX models for initialization of a and b
# ----------------------------------------------------------------------

result = minimize(fun=f_CMLE_arx, x0=theta_init, method='trust-constr', jac=g_CMLE_arx, hess=H_CMLE_arx, options={'verbose': 2,'maxiter':1000})
theta_arx = result.x.reshape(-1)


# ----------------------------------------------------------------------
#  ICMLE of ARMAX models
# ----------------------------------------------------------------------

theta_arx = jnp.hstack([theta_arx[:4],c_init,theta_arx[4:8],c_init,theta_arx[8:],c_init])

result = minimize(fun=f_CMLE, x0=theta_arx, method='trust-constr', jac=g_CMLE, hess=H_CMLE, options={'verbose': 2,'maxiter':1000})
theta_final = result.x.reshape(-1)




jnp.save("results/theta_opt_y1y2y3_arx_init_TR_given_xc",jnp.array(theta_final))
jnp.save("results/costs_opt_y1y2y3_arx_init_TR_given_xc",jnp.array(result.fun))
jnp.save("results/accuracy_y1y2y3_arx_init_TR_given_xc",jnp.array(result.success))
jnp.save("results/times_y1y2y3_arx_init_TR_given_xc",jnp.array(time.time() - startTime))
