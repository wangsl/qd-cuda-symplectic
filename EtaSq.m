
% Ref: J. Chem. Phys. 110, 11221 (1999)

function [ eta2 ] = EtaSq(R, E)

% E: translational energy

delta = R.delta;
k0 = R.k0;
mu = R.mass;

kE = sqrt(2*mu*E);

%eta = (pi)^(1/4)*(2*m*delta./kE).^(1/2).*exp(-delta*delta/2*(kE- ...
%						  k0).^2);
%eta2 = abs(eta).^2;

eta2 = pi^(1/2)*2*delta*mu./kE.*exp(-delta*delta*(kE-k0).^2);

return
