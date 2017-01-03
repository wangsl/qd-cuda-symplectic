
function [] = PlotCRP()

global H2eV 
global HO2Data

E = HO2Data.CRP.energies*H2eV;
CRP = -HO2Data.CRP.CRP;

persistent CRPPlot

if isempty(CRPPlot) 
  figure(2)
  CRPPlot = plot(E, CRP, 'b-', 'LineWidth', 2, 'YDataSource', ...
		 'CRP');
  
  set(gca, 'xtick', [0.4:0.2:2.2]);
  
  xlabel('Energy (eV)')
  ylabel('Reaction probabilities')
  
  axis([0.5 2.2 0 0.70])
  
  grid on
  grid minor
end

refreshdata(CRPPlot, 'caller');

drawnow


