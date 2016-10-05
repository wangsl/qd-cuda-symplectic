
% $Id$

function [ p ] = NormalizedLegendreP(n, x)

if n == 0
  p = ones(size(x));
elseif n == 1
  p = x;
else
  p0 = 1;
  p1 = x;
  for i = 2 : n
    p2 = (2-1/i)*x.*p1 - (1-1/i).*p0;
    p0 = p1;
    p1 = p2;
  end
  p = p2;
end

% Normailized Legendre polynomials

p = sqrt(n+1/2)*p;
