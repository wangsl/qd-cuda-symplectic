
% $Id$

function [] = PlotPotWave()

global HO2Data

persistent has_PotWavePlot
persistent hpsi

theta = HO2Data.theta;

r1 = HO2Data.r1;
r2 = HO2Data.r2;
pot = HO2Data.potential;

omega = 2;

k = 20;

psiReal = real(HO2Data.wavepacket_parameters.weighted_wavepackets(:, ...
						  :, k, omega))/sqrt(theta.w(k));
psiReal = psiReal';

if isempty(has_PotWavePlot)
  
  has_PotWavePlot = 1;
  
  figure(1);
  
  [ ~, hPES ] = contour(r1.r, r2.r, pot(:,:,k)', [ -0.2:0.01:0.3 ]);
  set(hPES, 'LineWidth', 0.75);
  set(hPES, 'LineColor', 'black');
  hold on;
  
  [ ~, hpsi ] = contour(r1.r, r2.r, psiReal, ...
			[ -2.0:0.02:-0.01 0.01:0.02:1.0 ], 'zDataSource', 'psiReal');
  set(hpsi, 'LineWidth', 1.5);
  set(gca, 'CLim', [-0.5, 0.5]);
  colormap jet
  colorbar('vert')
  hold off;
end

refreshdata(hpsi, 'caller');
drawnow
