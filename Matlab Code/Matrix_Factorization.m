
datastruct = importdata('data_nips.mat');
data = getfield(datastruct,'data_reformed');
vocab = getfield(datastruct,'vocab');
k = 10;

% C = Object-Object Matrix, C_ij = p(X1 = i, X2 = j), joint probability of
% pairs of words
% B = Object-Cluster Matrix, B_ik = p(X = i|Z = k), drawing word i given
% cluster k
% A = Cluster-Cluster Matrix, A_kl = p(Z1 = k, Z2 = l), joint probability
% of pairs of clusters

%% Generating Co-Occurance Matrix

%Call Q Matrix Function
Q = generateQMatrix(data);
Q = sparse(Q);
Q_Star = dykstra(Q,10);
