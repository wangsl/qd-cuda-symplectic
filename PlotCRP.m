
function [] = PlotCRP()

global H2eV 
global HO2Data

eO2 = HO2Data.CRP.eDiatomic;

%E = (HO2Data.CRP.energies - eO2)*H2eV;

E = HO2Data.CRP.energies*H2eV;
CRP = -HO2Data.CRP.CRP;
%CRP2 = -HO2Data.CRP.CRP*2*pi;

persistent CRPPlot
%persistent CRPPlot2

if isempty(CRPPlot) 
  figure(2)
  CRPPlot = plot(E, CRP, 'b-', 'LineWidth', 3, 'YDataSource', ...
		 'CRP');
end

%{
if(isempty(CRPPlot2))
  figure(3)
  CRPPlot2= plot(E, CRP2, 'g-', 'LineWidth', 3, 'YDataSource', 'CRP2');
end
%}

refreshdata(CRPPlot, 'caller');
%refreshdata(CRPPlot2, 'caller');
drawnow

