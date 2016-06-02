%NIPS:
%Docs = 1500
%Words = 12419
%Total Length without Headers = 746316
data = importdata('data/docword.nips.txt');
vocab = importdata('data/vocab.nips.txt');

no_docs = 1500;
vocab_size = 12419;

doc_no = data(1,1);
j = 1;
values = unique(data(:,2));
hist = histc(data(:,2),values);
word_ids_rem = values(find(hist < 5));
data_indices_rem = find(ismember(data(:,2),word_ids_rem));
data(data_indices_rem,:) = [];

data_reformed = zeros(no_docs,vocab_size);
for i = 1:size(data,1)
    if data(i,1) ~= doc_no
        j = j + 1;
        doc_no = data(i,1);
    end
    data_reformed(j,data(i,2)) = data(i,3);
end
clear data data_indices_rem doc_no hist i indices_to_remove j no_docs remove values vocab_size word_ids_rem;
save('data_nips.mat');