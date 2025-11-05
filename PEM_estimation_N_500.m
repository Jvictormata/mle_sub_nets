clear all
clc


N = 500;       

tolerance = 1e-5;

load('data/data_estimation.mat');

y1 = x(1:500);
y2 = x(501:1000);
y3 = x(1001:1500);

y6 = x(2501:3000);
r1 = double(r(:,1));
r2 = double(r(:,2));




%%%%%%%%%%%       X_o = (y1, y2, y3)        %%%%%%%%%%%%%%%


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


theta_pem_y1y2y3 = [estG1H1.A(2:3) estG1H1.B(2:3) estG1H1.C(2:3) estG2H2.A(2:3) estG2H2.B(2:3) estG2H2.C(2:3) estG3H3.A(2:3) estG3H3.B(2:3) estG3H3.C(2:3)  ];
save("results/theta_pem_y1y2y3.mat","theta_pem_y1y2y3")



%%%%%%%%%%%       X_o = (y1, y3)        %%%%%%%%%%%%%%%


Z1 = iddata(y1,[y6]);
na = [2]; 
nb = [2];
nc = [2];
nk = [1];
opt = armaxOptions;
opt.SearchOptions.Tolerance = tolerance;
estG1H1 = armax(Z1, [na nb nc nk],opt)


Z2 = iddata(y3,[r2 r1+y1+y3]);
nf = [2 4]; 
nb = [2 4];
nc = [4]; 
nd = [4]; 
nk = [1 1];
opt = bjOptions;
opt.SearchOptions.Tolerance = tolerance;
estG2G3 = bj(Z2, [nb nc nd nf nk],opt)



theta_pem_y1y3_G2G3 = [estG2G3.F{2}(2:5) estG2G3.B{2}(2:5)];

theta_pem_y1y3 = [estG1H1.A(2:3) estG1H1.B(2:3) estG1H1.C(2:3) 0 0 0 0 0 0 estG2G3.F{1}(2:3) estG2G3.B{1}(2:3) 0 0];
save("results/theta_pem_y1y3.mat","theta_pem_y1y3")
save("results/theta_pem_y1y3_G2G3.mat","theta_pem_y1y3_G2G3")


%%%%%%%%%%%       X_o = (y3)        %%%%%%%%%%%%%%%



Z2 = iddata(y3,[r2 r1+y1+y3 y6]);
nf = [2 4 6]; 
nb = [2 4 6];
nc = [6]; 
nd = [6]; 
nk = [1 1 1];
opt = bjOptions;
opt.SearchOptions.Tolerance = tolerance;
estG1G2G3 = bj(Z2, [nb nc nd nf nk],opt)



theta_pem_y3_G2G3 = [estG1G2G3.F{2}(2:5) estG1G2G3.B{2}(2:5)];
theta_pem_y3_G1G2G3 = [estG1G2G3.F{3}(2:7) estG1G2G3.B{3}(2:7)];

theta_pem_y3 = [0 0 0 0 0 0 0 0 0 0 0 0 estG1G2G3.F{1}(2:3) estG1G2G3.B{1}(2:3) 0 0];
save("results/theta_pem_y3.mat","theta_pem_y3")
save("results/theta_pem_y3_G2G3.mat","theta_pem_y3_G2G3")
save("results/theta_pem_y3_G1G2G3.mat","theta_pem_y3_G1G2G3")



