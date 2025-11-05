clear all
clc

load("data_estimation.mat")

N = 500;

load("true_system.mat");


sigma1 =  sqrt(0.01);
sigma2 =  sqrt(0.02);
sigma3 =  sqrt(0.03);
sigma4 =  sqrt(0.04);
sigma5 =  sqrt(0.05);
sigma6 =  sqrt(0.06);
sigma7 =  sqrt(0.07);


r1 =  [(0:N-1)'  r(:,1)];
r2 =  [(0:N-1)'  r(:,2)];
r3 =   [(0:N-1)'  r(:,3)];



x_100 = [];
for i=1:100
    e1 =  [(0:N-1)'  sigma1*randn(N,1)];
    e2 =  [(0:N-1)'  sigma2*randn(N,1)];
    e3 =  [(0:N-1)'  sigma3*randn(N,1)];
    e4=  [(0:N-1)'  sigma4*randn(N,1)];
    e5 =  [(0:N-1)'  sigma5*randn(N,1)];
    e6 =  [(0:N-1)'  sigma6*randn(N,1)];
    e7 =  [(0:N-1)'  sigma7*randn(N,1)];

out = sim('network_armax.slx',N-1);


y1 = out.y1;
y2 = out.y2;
y3 = out.y3;
y4 = out.y4;
y5 = out.y5;
y6 = out.y6;
y7 = out.y7;


u1 = y6;
u2 = out.r1+y1+y3;
u3 = y2+out.r2;
u4 = y7;
u5 = y4;
u6 = y5+y7+out.r3;
u7 = y3;

x = [y1(1:N);y2(1:N);y3(1:N);y4(1:N);y5(1:N);y6(1:N);y7(1:N);u1(1:N);u2(1:N);u3(1:N);u4(1:N);u5(1:N);u6(1:N);u7(1:N)];
x_100 = [x_100 x];
i
end

save("x_100_realizations.mat","x_100")