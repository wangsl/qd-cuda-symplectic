
close all
clear all
clc

x = linspace(-6, 6, 256);
y = linspace(-10, 10, 512);

phix = 1/pi.^(0.25)*exp(-0.5*x.*x);
phiy = sqrt(2.0)/pi.^(0.25)*y.*exp(-0.5*y.*y);

phi = phix'*phiy;

figure
contour(x, y, phi')

% [ X, Y ] = meshgrid(x, y);
% contour(X, Y, phi)

