
% this function will be replaced with fcnpak later
% http://math.nist.gov/~DLozier/projects/FCNPAK/fcnpak

function [ P ] = AssLegendreP(Omega, lMax, x)

assert(lMax >= Omega)

nL = lMax - Omega + 1;
P = zeros(numel(x), nL);

for l = Omega : lMax
  p = legendre(l, x, 'norm');
  P(:, l-Omega+1) = p(Omega+1, :)';
end

return
