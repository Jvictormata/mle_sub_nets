clear all
clc


N = 500;
load("data/data_validation.mat");
load("data/true_system.mat")

true_theta = theta;

load("results/theta_pem_y1y2y3.mat");
load("results/theta_pem_y1y3.mat");
load("results/theta_pem_y1y3_G2G3.mat");
load("results/theta_pem_y3.mat");
load("results/theta_pem_y3_G2G3.mat");
load("results/theta_pem_y3_G1G2G3.mat");


pyData = py.numpy.load("results/theta_opt_y3_arx_init_TR.npy");  
theta_mle_y3= double(pyData);
pyData = py.numpy.load("results/times_y3_arx_init_TR.npy");   
time_mle_y3 = double(pyData);

pyData = py.numpy.load("results/theta_opt_y1y3_arx_init_TR.npy");   
theta_mle_y1y3= double(pyData);
pyData = py.numpy.load("results/times_y1y3_arx_init_TR.npy");   
time_mle_y1y3 = double(pyData);

pyData = py.numpy.load("results/theta_opt_y1y2y3_arx_init_TR.npy");  %theta_opt_y1y2y3_arx_init_TR_given_xc
theta_mle_y1y2y3= double(pyData);
pyData = py.numpy.load("results/times_y1y2y3_arx_init_TR.npy");   
time_mle_y1y2y3 = double(pyData);


fits_xo_mle_y3_sim = [];
fits_xo_mle_y1y3_sim = [];
fits_xo_mle_y1y2y3_sim = [];


order_barH = [];




%%%%% Getting the validation data:

r1 =  [(0:N-1)' r(:,1) ];
r2 =  [(0:N-1)' r(:,2) ];



%%%%%%%% Real system: %%%%%%%%


y1r = x(1:500);
y2r = x(501:1000);
y3r = x(1001:1500);
y4r = x(1501:2000);
y5r = x(2001:2500);
y6r = x(2501:3000);
y7r = x(3001:3500);

u1r = x(3501:4000);
u2r = x(4001:4500);
u3r = x(4501:5000);
u4r = x(5001:5500);
u5r = x(5501:6000);
u6r = x(6001:6500);
u7r = x(6501:7000);

r_y6 =  [(0:N-1)'  y6r];



%%%%%%%% Estimated system PEM - yo = y1,y2,y3: %%%%%%%%


theta = theta_pem_y1y2y3;


out = sim('data/validation_network_armax_y1y2y3.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;


fit_xo_pem_y1y2y3_y1 = 1 - norm([y1]-[y1r])/norm([y1r]-mean([y1r]))
fit_xo_pem_y1y2y3_y2 = 1 - norm([y2]-[y2r])/norm([y2r]-mean([y2r]))
fit_xo_pem_y1y2y3_y3 = 1 - norm([y3]-[y3r])/norm([y3r]-mean([y3r]))



%%%%%%%% Estimated system PEM - yo = y1,y3: %%%%%%%%

theta = theta_pem_y1y3;
theta_2 = theta_pem_y1y3_G2G3;


out = sim('data/validation_network_armax_y1y3.slx',N-1);

y1 = out.y1;
y3 = out.y3;


fit_xo_pem_y1y3_y1 = 1 - norm([y1]-[y1r])/norm([y1r]-mean([y1r]))
fit_xo_pem_y1y3_y3 = 1 - norm([y3]-[y3r])/norm([y3r]-mean([y3r]))


%%%%%%%% Estimated system PEM - yo = y3: %%%%%%%%

theta = theta_pem_y3;
theta_2 = theta_pem_y3_G2G3;
theta_3 = theta_pem_y3_G1G2G3;


out = sim('data/validation_network_armax_y3.slx',N-1);

y3 = out.y3;



fit_xo_pem_y3_y3 = 1 - norm([y3]-[y3r])/norm([y3r]-mean([y3r]))




%%%%%%%% Estimated system MLE - x_o = y3: %%%%%%%%
theta = theta_mle_y3(1:18);

% Simulation Data:

out = sim('data/validation_network_armax_y1y2y3.slx',N-1);

y1 = out.y1;
y2 = out.y2;
y3 = out.y3;


fit_xo_mle_y3_y1 = 1 - norm([y1]-[y1r])/norm([y1r]-mean([y1r]))
fit_xo_mle_y3_y2 = 1 - norm([y2]-[y2r])/norm([y2r]-mean([y2r]))
fit_xo_mle_y3_y3 = 1 - norm([y3]-[y3r])/norm([y3r]-mean([y3r]))



%%%%%%%% Estimated system MLE - x_o = y1y3: %%%%%%%%
theta =  theta_mle_y1y3(1:18);

% Simulation Data:

out = sim('data/validation_network_armax_y1y2y3.slx',N-1);


y1 = out.y1;
y2 = out.y2;
y3 = out.y3;


fit_xo_mle_y1y3_y1 = 1 - norm([y1]-[y1r])/norm([y1r]-mean([y1r]))
fit_xo_mle_y1y3_y2 = 1 - norm([y2]-[y2r])/norm([y2r]-mean([y2r]))
fit_xo_mle_y1y3_y3 = 1 - norm([y3]-[y3r])/norm([y3r]-mean([y3r]))



%%%%%%%% Estimated system MLE - x_o = y1y2y3: %%%%%%%%
theta =  theta_mle_y1y2y3(1:18);

% Simulation Data:

out = sim('data/validation_network_armax_y1y2y3.slx',N-1);


y1 = out.y1;
y2 = out.y2;
y3 = out.y3;


fit_xo_mle_y1y2y3_y1 = 1 - norm([y1]-[y1r])/norm([y1r]-mean([y1r]))
fit_xo_mle_y1y2y3_y2 = 1 - norm([y2]-[y2r])/norm([y2r]-mean([y2r]))
fit_xo_mle_y1y2y3_y3 = 1 - norm([y3]-[y3r])/norm([y3r]-mean([y3r]))


