function [ Q ] = generateQMatrix( docMat)
%generateQMatrix Given a sparse matrix docMat, it computes word-word
%co-occurences Q and returns it
%   docMat - Matrix of document x word (size: (docs,vocab))
%   Q - word-word co-occurance
%   numDocs,vocabSize - # of docs, vocabulary size
%   Nm - Total words per document

numDocs = size(docMat,1);
vocabSize = size(docMat,2);
diag_docMat = zeros(1,vocabSize);

for i = 1:numDocs
    Nm = sum(docMat(i,:));
    diag_docMat = diag_docMat + (docMat(i,:) / (Nm * (Nm - 1)));
    docMat(i,:) = docMat(i,:) / sqrt((Nm * (Nm - 1)));
end

Q = docMat' * docMat /numDocs;
diag_docMat = diag_docMat / numDocs;
Q = Q - diag(diag_docMat);

end

