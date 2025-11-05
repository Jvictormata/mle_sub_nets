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


#Interconections
Lamb = jnp.array([[0,0,0],[1,0,1],[0,1,0]])
Delta = jnp.eye(3)


########## Loading data:

data = loadmat('data/data_estimation.mat')


y6 = data["x"][2500:3000]
r = jnp.block([y6.reshape(-1,1), data["r"][:,0].reshape(-1,1), data["r"][:,1].reshape(-1,1)])
x = jnp.concatenate([data["x"][0:1500], data["x"][3500:5000]]).reshape(-1,1)

## Initial theta
theta_init = 0.1*jnp.ones(12)
c_init = jnp.array([0.1, 0.1])
n_ab_arx = ((2,2,2),(2,2,2))




r_sig = jnp.concatenate([r[:,0],r[:,1],r[:,2]]).reshape(-1,1)

xo,_,Permut = get_xoxm(x,obs,N,M)

A2,B2 = gen_A2B2(Lamb,Delta,Permut,M,N,r_sig)
A2o = A2[:,:n_obs*N]
A2m = A2[:,n_obs*N:]
To_phi, Tm_phi, T_gamma, W2, V2 = get_transform_matrices(A2o,A2m,B2)


@jit
def f_MLE_arx(theta):
    f = eval_cost_func(theta.reshape(-1),n_ab_arx,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
    f = jnp.where(jnp.isnan(f), 1.0e13, f)
    return f


@jit
def g_MLE_arx(theta):
    return grad_theta(theta.reshape(-1),n_ab_arx,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(-1)


@jit
def H_MLE_arx(theta):
    return Hessian_theta(theta.reshape(-1),n_ab_arx,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])


@jit
def f_MLE(theta):
    f = eval_cost_func(theta.reshape(-1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
    f = jnp.where(jnp.isnan(f), 1.0e13, f)
    return f


@jit
def g_MLE(theta):
    return grad_theta(theta.reshape(-1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(-1)


@jit
def H_MLE(theta):
    return Hessian_theta(theta.reshape(-1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])


@jit
def f_MLE_dif_lambdas(theta):
    f = eval_cost_func_dif_lambda(theta.reshape(-1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
    f = jnp.where(jnp.isnan(f), 1.0e13, f)
    return f

@jit
def g_MLE_dif_lambdas(theta):
    return grad_theta_dif_lambda(theta.reshape(-1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(-1)

@jit
def H_MLE_dif_lambdas(theta):
    return Hessian_theta_dif_lambda(theta.reshape(-1),n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo).reshape(theta.shape[0],theta.shape[0])


startTime = time.time()

# ----------------------------------------------------------------------
#  MLE of ARX models for initialization of a and b
# ----------------------------------------------------------------------

result = minimize(fun=f_MLE_arx, x0=theta_init, method='trust-constr', jac=g_MLE_arx, hess=H_MLE_arx, options={'verbose': 2,'maxiter':1000})
theta_arx = result.x.reshape(-1)


# ----------------------------------------------------------------------
#  Initialization of c and lambda - MLE - same lambda for all e 
# ----------------------------------------------------------------------

theta_arx = jnp.hstack([theta_arx[:4],c_init,theta_arx[4:8],c_init,theta_arx[8:],c_init])

result = minimize(fun=f_MLE, x0=theta_arx, method='trust-constr', jac=g_MLE, hess=H_MLE, options={'verbose': 2,'maxiter':1000})
theta_same_lambda = result.x.reshape(-1)

lambda_same = get_same_lambda(theta_same_lambda,n_ab,Permut,N,M,To_phi, Tm_phi, T_gamma, W2, xo)
lambda_init = -(1/2)*jnp.log10(lambda_same) # λ_i^{-½} = 10^{θ_i}, i = -M:end  -> θ_i = -0.5*log10(λ_i)


# ----------------------------------------------------------------------
#  MLE  - different lambdas
# ----------------------------------------------------------------------


result = minimize(fun=f_MLE_dif_lambdas, x0= jnp.hstack([theta_same_lambda,lambda_init,lambda_init,lambda_init]), method='trust-constr', jac=g_MLE_dif_lambdas,hess=H_MLE_dif_lambdas, options={'verbose': 2,'maxiter':1000})
theta_final = result.x.reshape(-1)


jnp.save("results/theta_opt_y1y2y3_arx_init_TR",jnp.array(theta_final))
jnp.save("results/costs_opt_y1y2y3_arx_init_TR",jnp.array(result.fun))
jnp.save("results/accuracy_y1y2y3_arx_init_TR",jnp.array(result.success))
jnp.save("results/times_y1y2y3_arx_init_TR",jnp.array(time.time() - startTime))
