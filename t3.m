
close all
clear all

n = 16;
 
[ x, w ] = GaussLegendreGrids(n);

OmegaMin = 1;
OmegaMax = 3;
lMax = 10;

tic
P = AssociatedLegendreP(OmegaMin, OmegaMax, lMax, x);
toc

return;

for k = 1 : n
  P(k, :, :) = P(k, :, :)*sqrt(w(k));
end

Om = 32;

iOffset = 1 - OmegaMin;

Om = Om + iOffset;

for l1 = Om-4 : Om+5
  for l2 = Om-4 : Om+5
    fprintf(' %4d %4d  %16.12f\n', l1-iOffset, l2-iOffset, sum(P(:, l1, Om).*P(:, l2, Om)))
  end
end
