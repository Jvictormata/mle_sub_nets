clear all
clc


N= 5000;

e1 =  [(0:N-1)' linspace(1,1,N)' ];
e2 =  [(0:N-1)' linspace(1,1,N)' ];
e3 =  [(0:N-1)' linspace(1,1,N)' ];
e4 =  [(0:N-1)' linspace(1,1,N)' ];
e5 =  [(0:N-1)' linspace(1,1,N)' ];
e6 =  [(0:N-1)' linspace(1,1,N)' ];
e7 =  [(0:N-1)' linspace(1,1,N)' ];


r1 =  [(0:N-1)' linspace(1,1,N)' ];
r2 =  [(0:N-1)' linspace(1,1,N)' ];
r3 =  [(0:N-1)' linspace(1,1,N)' ];
r4 =  [(0:N-1)' linspace(1,1,N)' ];


load("true_system.mat")