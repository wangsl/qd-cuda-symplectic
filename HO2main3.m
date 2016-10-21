

%function [] = HO2main(jRot, nVib)

clear all
clc

format long

%if nargin == 0 
jRot = 1;
nVib = 0;
%end

%setenv('OMP_NUM_THREADS', '16');

global H2eV 
global HO2Data

H2eV = 27.21138505;

MassAU = 1.822888484929367e+03;

mH = 1.0079;
mO = 15.999;

masses = [ mH, mO, mO ];

masses = masses*MassAU;

% time

time.total_steps = int32(1000000);
time.time_step = 1.0; %0.5;
time.steps = int32(0);

% r1: R

r1.n = int32(256);
r1.r = linspace(1.6, 16.0, r1.n);
r1.left = r1.r(1);
r1.dr = r1.r(2) - r1.r(1);
r1.mass = masses(1)*(masses(2)+masses(3))/(masses(1)+masses(2)+ ...
					   masses(3));
r1.dump = WoodsSaxon(4.0, 14.5, r1.r);

r1.r0 = 10.0;
r1.k0 = 0.35;
r1.delta = 0.06;

%{
r1.r0 = 10.0;
r1.k0 = 11.9366;
r1.delta = 0.6;
%}

eGT = 1/(2*r1.mass)*(r1.k0^2 + 1/(2*r1.delta^2));
fprintf(' Gaussian wavepacket kinetic energy: %.15f\n', eGT)

% r2: r

r2.n = int32(256);
r2.r = linspace(1.6, 12.0, r2.n);
r2.left = r2.r(1);
r2.dr = r2.r(2) - r2.r(1);
r2.mass = masses(2)*masses(3)/(masses(2)+masses(3));

r2.dump = WoodsSaxon(4.0, 10.0, r2.r);

% dividing surface

rd = 7.0;
nDivdSurf = int32((rd - min(r2.r))/r2.dr);
r2Div = double(nDivdSurf)*r2.dr + min(r2.r);
fprintf(' Dviding surface: %.15f\n', r2Div);

% theta

theta.n = int32(180);
[ theta.x, theta.w ] = GaussLegendreGrids(theta.n);

%theta.legendre = LegendreP2(double(theta.m), theta.x);
% transpose Legendre polynomials in order to do 
% matrix multiplication in C++ and Fortran LegTransform.F
%theta.legendre = theta.legendre';

% options

options.wave_to_matlab = 'HO2Matlab';
options.CRPMatFile = sprintf('CRPMat-j%d-v%d.mat', jRot, nVib);
options.steps_to_copy_psi_from_device_to_host = int32(100);
options.potential_cutoff = 4.0;

% setup potential energy surface and initial wavepacket
potential = DMBEIVPESJacobi(r1.r, r2.r, theta.x, masses);

[ psi, eO2 ] = InitWavePacket(r1, r2, theta, jRot, nVib);

% PlotPotWave(r1, r2, potential, psi)

J = 3;
parity = 0;
lMax = 120;

wavepacket_parameters.J = int32(J);
wavepacket_parameters.parity = int32(parity);
wavepacket_parameters.lMax = int32(lMax);

[ OmegaMin, OmegaMax ] = OmegaRange(J, parity, lMax);

wavepacket_parameters.OmegaMin = int32(OmegaMin);
wavepacket_parameters.OmegaMax = int32(OmegaMax);

P = AssociatedLegendreP(OmegaMin, OmegaMax, lMax, theta.x);
for k = 1 : theta.n
  P(k,:,:) = P(k,:,:)*sqrt(theta.w(k));
end

wavepacket_parameters.weighted_associated_legendres = P;

nOmegas = OmegaMax - OmegaMin + 1;
wavepackets = zeros([size(psi), nOmegas]);
wavepackets(:,:,:,2) = psi;
%for o = 1 : nOmegas
%  wavepackets(:,:,:,o) = psi;
%end
for k = 1 : theta.n
  wavepackets(:,:,k,:) = wavepackets(:,:,k,:)*sqrt(theta.w(k));
end

%{
sum(sum(sum(conj(wavepackets).*wavepackets)))*r1.dr*r2.dr

sum(sum(sum(conj(wavepackets(:,:,:,1)).*...
		 wavepackets(:,:,:,1))))*r1.dr*r2.dr


sum(sum(sum(conj(wavepackets(:,:,:,1)).*potential.*...
		 wavepackets(:,:,:,1))))*r1.dr*r2.dr
%}

%%{
for o = OmegaMin : OmegaMax
  O = o + 1 - OmegaMin;
  p1 = P(:,:,O);
  psi1 = wavepackets(:,:,:,O);
  [ n1, n2, n3 ] = size(psi1);
  psi1 = reshape(psi1, [n1*n2, n3]);
  g1 = psi1*p1;
  psi1 = g1*p1';
  s = sum(sum(conj(psi1).*psi1))*r1.dr*r2.dr;
  fprintf(' %d %.15f\n', O, s)
end
clear p1 psi1 n1 n2 n3 whog1 s o O
%%}

wavepacket_parameters.weighted_wavepackets = wavepackets;

% cummulative reaction probabilities

%CRP.eDiatomic = eO2;
%CRP.n_dividing_surface = nDivdSurf;
%CRP.n_gradient_points = int32(31);
%CRP.n_energies = int32(300);
%eLeft = 0.6/H2eV + eO2;
%eRight = 1.8/H2eV + eO2;
%CRP.energies = linspace(eLeft, eRight, CRP.n_energies);
%CRP.eta_sq = EtaSq(r1, CRP.energies-eO2);
%CRP.CRP = zeros(size(CRP.energies));
%CRP.calculate_CRP = int32(1);

% pack data to one structure

HO2Data.r1 = r1;
HO2Data.r2 = r2;
HO2Data.theta = theta;
HO2Data.potential = potential;
%HO2Data.psi = psi;
HO2Data.time = time;
HO2Data.options = options;
%HO2Data.CRP = CRP;

HO2Data.wavepacket_parameters = wavepacket_parameters;

PlotPotWave();
%return

%clearvars -except HO2Data
%whos

% time evolution

tic
cudaSymplectic(HO2Data);
toc

return

