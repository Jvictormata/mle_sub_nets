clear all
clc

load("data/true_system.mat");
real_theta = theta(1:18)';

N = 500;
N_orig = 500; 

%%%% MLE  y1 y2 y3
pyData = py.numpy.load("results/cov/theta_opt_y1y2y3_100_rel_arx_init_TR.npy"); 
thetas_mle_y1y2y3_100_rel = double(pyData);
pyData = py.numpy.load("results/cov/accuracy_y1y2y3_100_rel_arx_init_TR.npy"); 
accuracy_y1y2y3_100 = double(pyData);
thetas_mle_y1y2y3_100_rel = thetas_mle_y1y2y3_100_rel(logical(accuracy_y1y2y3_100), :);


thetas_mle_y1y2y3_100_rel = thetas_mle_y1y2y3_100_rel';
thetas_mle_y1y2y3_100_rel = thetas_mle_y1y2y3_100_rel(1:18,:);

%%%% MLE  y1 y3
pyData = py.numpy.load("results/cov/theta_opt_y1y3_100_rel_arx_init_TR.npy");  
thetas_mle_y1y3_100_rel = double(pyData);
pyData = py.numpy.load("results/cov/accuracy_y1y3_100_rel_arx_init_TR.npy"); 
accuracy_y1y3_100 = double(pyData);
thetas_mle_y1y3_100_rel = thetas_mle_y1y3_100_rel(logical(accuracy_y1y3_100), :);


thetas_mle_y1y3_100_rel = thetas_mle_y1y3_100_rel';
thetas_mle_y1y3_100_rel = thetas_mle_y1y3_100_rel(1:18,:);

%%%% MLE  y3
pyData = py.numpy.load("results/cov/theta_opt_y3_100_rel_arx_init_TR.npy");  
thetas_mle_y3_100_rel = double(pyData);
pyData = py.numpy.load("results/cov/accuracy_y3_100_rel_arx_init_TR.npy"); 
accuracy_y3_100 = double(pyData);
thetas_mle_y3_100_rel = thetas_mle_y3_100_rel(logical(accuracy_y3_100), :);

thetas_mle_y3_100_rel = thetas_mle_y3_100_rel';
thetas_mle_y3_100_rel = thetas_mle_y3_100_rel(1:18,:);


% Considering only a,b:
real_theta(5:6) = 0;
real_theta(11:12) = 0;
real_theta(17:18) = 0;

thetas_mle_y1y2y3_100_rel(5:6,:) = 0;
thetas_mle_y1y2y3_100_rel(11:12,:) = 0;
thetas_mle_y1y2y3_100_rel(17:18,:) = 0;

thetas_mle_y1y3_100_rel(5:6,:) = 0;
thetas_mle_y1y3_100_rel(11:12,:) = 0;
thetas_mle_y1y3_100_rel(17:18,:) = 0;

thetas_mle_y3_100_rel(5:6,:) = 0;
thetas_mle_y3_100_rel(11:12,:) = 0;
thetas_mle_y3_100_rel(17:18,:) = 0;



%%%%%%%% Computin Covariance - PEM - xo = (y1,y2,y3): %%%%%%%%
load('data/data_estimation.mat');
load('data/x_100_realizations.mat');

tolerance = 1e-5;

r1 = double(r(:,1));
r2 = double(r(:,2));

thetas_pem_100 = [];

for i =1:100


y1 = x_100(1:500,i);
y2 = x_100(501:1000,i);
y3 = x_100(1001:1500,i);

y6 = x_100(2501:3000,i);


Z1 = iddata(y1,[y6]);
na = [2]; 
nb = [2];
nc = [2];
nk = [1];
opt = armaxOptions;
opt.SearchOptions.Tolerance = tolerance;
estG1H1 = armax(Z1, [na nb nc nk],opt)

Z2 = iddata(y2,[r1+y1+y3]);
na = [2]; 
nb = [2];
nc = [2];
nk = [1];
opt = armaxOptions;
opt.SearchOptions.Tolerance = tolerance;
estG2H2 = armax(Z2, [na nb nc nk],opt)


Z3 = iddata(y3,[r2+y2]);
na = [2]; 
nb = [2];
nc = [2];
nk = [1];
opt = armaxOptions;
opt.SearchOptions.Tolerance = tolerance;
estG3H3 = armax(Z3, [na nb nc nk],opt)


theta_pem_y1y2y3 = [estG1H1.A(2:3) estG1H1.B(2:3) 0 0 estG2H2.A(2:3) estG2H2.B(2:3) 0 0 estG3H3.A(2:3) estG3H3.B(2:3) 0 0 ];
thetas_pem_100 = [thetas_pem_100; theta_pem_y1y2y3];
end


thetas_pem_100 = thetas_pem_100';
thetas_pem_100_orig = thetas_pem_100;

exp_theta_pem_y1y2y3 = mean(thetas_pem_100,2);

cov_pem_y1y2y3 = zeros(size(exp_theta_pem_y1y2y3,1),size(exp_theta_pem_y1y2y3,1));

difference = thetas_pem_100 - exp_theta_pem_y1y2y3;
for i=1:100
    cov_pem_y1y2y3 = cov_pem_y1y2y3 + difference(:,i)*(difference(:,i)');
end

cov_pem_y1y2y3 = cov_pem_y1y2y3/100;
result_cov_pem = [trace(cov_pem_y1y2y3),max(eig(cov_pem_y1y2y3))]

bias_pem_y1y2y3 =  norm(exp_theta_pem_y1y2y3 - real_theta)^2

mse_pem_y1y2y3 = trace(cov_pem_y1y2y3) + norm(exp_theta_pem_y1y2y3 - real_theta)^2



%%%%%%%% Computing Covariance - MLE - xo = (y1,y2,y3): %%%%%%%%

% Cov:
exp_theta_mle_y1y2y3 = mean(thetas_mle_y1y2y3_100_rel,2);

difference = thetas_mle_y1y2y3_100_rel-exp_theta_mle_y1y2y3;

cov_mle_y1y2y3 = zeros(size(thetas_mle_y1y2y3_100_rel,1),size(thetas_mle_y1y2y3_100_rel,1));

for i=1:size(thetas_mle_y1y2y3_100_rel,2)
    cov_mle_y1y2y3 = cov_mle_y1y2y3 + difference(:,i)*(difference(:,i)');
end

cov_mle_y1y2y3 = cov_mle_y1y2y3/size(thetas_mle_y1y2y3_100_rel,2);
result_cov_mle_y1y2y3 =  [trace(cov_mle_y1y2y3) max(eig(cov_mle_y1y2y3))]

bias_mle_y1y2y3  = norm(exp_theta_mle_y1y2y3 - real_theta)^2
mse_mle_y1y2y3  = norm(exp_theta_mle_y1y2y3 - real_theta)^2 + trace(cov_mle_y1y2y3)




%%%%%%%% Computing Covariance - MLE - xo =  (y1,y3): %%%%%%%%

exp_theta_mle_y1y3 = mean(thetas_mle_y1y3_100_rel,2);

difference = thetas_mle_y1y3_100_rel-exp_theta_mle_y1y3;

cov_mle_y1y3 = zeros(size(thetas_mle_y1y3_100_rel,1),size(thetas_mle_y1y3_100_rel,1));

for i=1:size(thetas_mle_y1y3_100_rel,2)
    cov_mle_y1y3 = cov_mle_y1y3 + difference(:,i)*(difference(:,i)');
end

cov_mle_y1y3 = cov_mle_y1y3/size(thetas_mle_y1y3_100_rel,2);
result_cov_mle_y1y3 =  [trace(cov_mle_y1y3) max(eig(cov_mle_y1y3))]

bias_mle_y1y3  = norm(exp_theta_mle_y1y3 - real_theta)^2
mse_mle_y1y3  = norm(exp_theta_mle_y1y3 - real_theta)^2 + trace(cov_mle_y1y3)



%%%%%%%% Computing Covariance - MLE - xo =  (y3): %%%%%%%%

exp_theta_mle_y3 = mean(thetas_mle_y3_100_rel,2);

difference = thetas_mle_y3_100_rel-exp_theta_mle_y3;

cov_mle_y3 = zeros(size(thetas_mle_y3_100_rel,1),size(thetas_mle_y3_100_rel,1));

for i=1:size(thetas_mle_y3_100_rel,2)
    cov_mle_y3 = cov_mle_y3 + difference(:,i)*(difference(:,i)');
end

cov_mle_y3 = cov_mle_y3/size(thetas_mle_y3_100_rel,2);
result_cov_mle_y3 =  [trace(cov_mle_y3) max(eig(cov_mle_y3))]

bias_mle_y3  = norm(exp_theta_mle_y3 - real_theta)^2
mse_mle_y3  = norm(exp_theta_mle_y3 - real_theta)^2 + trace(cov_mle_y3)

