
clear all

H2eV = 27.21138505;

nMax = 20;

matFiles = cell(1, nMax);
energies = cell(1, nMax);
CRPs = cell(1, nMax);

n = 0;
for i = 2 : nMax
  index = strcat('00', int2str(i));
  index = index(end-1:end);
  matFile = strcat('CRPMat-j1-v0-', index, '.mat');
  if exist(matFile, 'file') == 2
    fprintf('%s\n', matFile)
    n = n + 1;
    load(matFile)
    energies{n} = CRP.energies*H2eV;
    CRPs{n} = CRP.CRP;
    matFiles{n} = matFile;
  end
end

for i = 1 : n
  plot(energies{i}, -CRPs{i}, 'LineWidth', 0.5, 'DisplayName', matFiles{i})
  hold on
end

legend('show', 'Location','northwest'); %,'Orientation','horizontal')

hold off

set(gca, 'xtick', [0.4:0.2:2.2]);

xlabel('Energy (eV)')
ylabel('Reaction probabilities')

axis([0.7 2.2 0 0.70])

pbaspect([1 0.5 1])

set(gca,'XMinorTick','on','YMinorTick','on')

grid on
grid minor

print -dpdf CRPv0j1.pdf

return

load('CRPMat-j1-v0-2.mat')
e2 = CRP.energies*H2eV;
CRP2 = CRP.CRP;

load('CRPMat-j1-v0-3.mat')
e3 = CRP.energies*H2eV;
CRP3 = CRP.CRP;

load('CRPMat-j1-v0-4.mat')
e4 = CRP.energies*H2eV;
CRP4 = CRP.CRP;

load('CRPMat-j1-v0-5.mat')
e5 = CRP.energies*H2eV;
CRP5 = CRP.CRP;

plot(e2, -CRP2, 'k', ...
     e3, -CRP3, 'b', ...
     e4, -CRP4, 'g', ...
     e5, -CRP5, 'r', ...
     'LineWidth', 1)

%plot(e2, -CRP2, 'b', 'LineWidth', 1)

set(gca, 'xtick', [0.4:0.2:2.2]);

xlabel('Energy (eV)')
ylabel('Reaction probabilities')

axis([0.7 2.2 0 0.70])

%axis([0.8 1.8 0 0.60])

pbaspect([1 0.5 1])

set(gca,'XMinorTick','on','YMinorTick','on')

grid on
grid minor

print -dpdf CRPv0j1.pdf

