
close all
clear all
clc

format long

psi = 0.1*pi*complex(1:20, 5:24);

size(psi)

%psi = reshape(psi, [4 5]);

data.psi = psi;

cudaSymplectic(data)

psi

