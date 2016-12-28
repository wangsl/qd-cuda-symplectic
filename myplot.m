
clear all

H2eV = 27.21138505;


%{
load('CRPMat-j1-v0-all.mat')
eAll = CRP.energies*H2eV;
CRPAll = -2*CRP.CRP;

load('CRPMat-j1-v0-odd.mat')
eOdd = CRP.energies*H2eV;
CRPOdd = -2*CRP.CRP;

plot(eAll, CRPAll, 'r', eOdd, CRPOdd, 'b', 'LineWidth', 2)
%}

load('CRPMat-j1-v0.mat')
eAll = CRP.energies*H2eV;
CRPAll = CRP.CRP;

plot(eAll, -CRPAll, 'b', 'LineWidth', 2)

set(gca, 'xtick', [0.4:0.2:2.2]);

xlabel('Energy (eV)')
ylabel('Reaction probabilities')

axis([0.5 2.2 0 0.70])

%axis([0.8 1.45 0 0.40])

grid on
grid minor

print -dpdf CRP.pdf

