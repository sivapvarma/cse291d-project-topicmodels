function [ X0 ] = dykstra(X, k)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

T = 10;
X0 = X;
P1 = zeros(size(X));
P2 = zeros(size(X));
P3 = zeros(size(X));
for i = 1:T
    disp(i);
    disp('Proj PSD');
    X1 = proj_psd(X0 + P1, k); 
	P1 = X0 + P1 - X1;
    disp('Proj Norm');
	X2 = proj_norm(X1 + P2);
	P2 = X1 + P2 - X2;
    disp('Proj NN');
	X0 = proj_nn(X2 + P3);
	P3 = X2 + P3 - X0 ;
end


end

function [X] = proj_psd(A, k)
    A = sparse(A);
    [eigvecs, eigvals] = eigs(A);
    [sortvals sortidxs] = sort(eigvals,'descend');
    eigvals(eigvals < eigvals(sortidxs(k))) = 0;
    X = eigvecs * eigvals * eigvecs';
end

function [Norm] = proj_norm(A)
    A = sparse(A);
    Norm = A + ((1 - sum(sum(A)))/(size(A,1) ^ 2));
end

function [NN] = proj_nn(A)
  NN = sparse(A);
  NN(NN < 0) = 0;
end