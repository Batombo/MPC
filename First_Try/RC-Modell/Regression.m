clc;
clear;
close all;
% theta = [35, 30, 25, 20, 15, 10, 5, 0];
% rho = [1.1455, 1.1644, 1.1839, 1.2041, 1.2250, 1.2466, 1.2690, 1.2922];

theta = [1,2,3, 4];
rho = [6.83, 59.61, 226.74, 607.075];

n = 3;
p = polyfit(theta, rho, n);

x = 0:0.1:5;

y = p(1)*x.^3 +  p(2)*x.^2 + p(3)*x.^1 + p(4);

plot(x,y)
hold on;
plot(theta, rho, 'r*')
hold off;

rho_reg =  p(1)*theta.^2 + p(2)*theta + p(3);

rms(rho_reg - rho)