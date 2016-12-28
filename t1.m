
sum(sum(sum(conj(wavepackets).*wavepackets)))*r1.dr*r2.dr

%n1 = r1.n;
%n2 = r2.n;
%nTheta = theta.n;

for o = OmegaMin : OmegaMax
  Omega = o + 1 - OmegaMin;
  p1 = P(:,:,Omega);
  %p1 = myP{o-OmegaMin+1}
  psi1 = wavepackets(:,:,:,Omega);
  [ n1, n2, n3 ] = size(psi1);
  psi1 = reshape(psi1, [n1*n2, n3]);
  g1 = psi1*p1;
  psi1 = g1*p1';
  s = sum(sum(conj(psi1).*psi1))*r1.dr*r2.dr;
  fprintf(' %d %.15f\n', Omega, s)
end

fprintf(' Odd Legendre test\n')

for o = OmegaMin : OmegaMax
  Omega = o + 1 - OmegaMin;
  %p1 = P(:,:,Omega);
  %p1 = myP{o-OmegaMin+1};
  p1 = myP{1};
  psi1 = wavepackets(:,:,:,Omega);
  sum(sum(sum(conj(psi1).*psi1)))*r1.dr*r2.dr
  [ n1, n2, n3 ] = size(psi1);
  psi1 = reshape(psi1, [n1*n2, n3]);
  sum(sum(conj(psi1).*psi1))*r1.dr*r2.dr
  g1 = psi1*p1;
  psi1 = g1*p1';
  s = sum(sum(conj(psi1).*psi1))*r1.dr*r2.dr;
  fprintf(' %d %.15f\n', Omega, s)
end


%clear p1 psi1 n1 n2 n3 whog1 s o Omega




