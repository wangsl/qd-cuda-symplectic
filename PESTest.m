
close all
clear all
clc

MassAU = 1.822888484929367e+03;

mH = 1.0079;
mO = 15.999;

masses = [ mH, mO, mO ];

masses = masses*MassAU;

R = linspace(0.3, 12.0, 256);
r = linspace(0.5, 12.0, 256);

n = 4;
m = 4;

cosThetas = linspace(-1.0, 0.0, n*m);

k = 0;
for i = 1 : n
  for j = 1 : m
    k = k + 1
    subplot(n, m, k)
    
    V = DMBEIVPESJacobi(R, r, cosThetas(k), masses);
    
    [ ~, hPES ] = contour(R, r, V', [ -0.2:0.01:1.0 ]);
    set(hPES, 'LineWidth', 0.05);
    %set(hPES, 'LineColor', 'black');
    hold on;
    
  end
end

print -dpdf PES-gamma.pdf

