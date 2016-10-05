
close all
clear all
clc

n = 3;

IND = 1:n*n;
s = [n, n];
[ I, J ] = ind2sub(s, IND);

a = 1./(I-J).^2 - 1./(I+J).^2;

a = reshape(a, [n, n])

b = zeros(n);

for i = 1 : n
  for j = 1 : n
    b(i,j) = 1/(i-j)^2 - 1/(i+j)^2;
  end
end

b



