# datafiles
vf_name = "../data/vocab.nips.txt"
dwf_name = "../data/docword.nips.txt"

function read_vocab(fname)
    fh = open(fname)
    v =  []
    for ln in eachline(fh)
        push!(v, strip(ln))
    end
    close(fh)
    return v
end

function read_word_doc_matrix(fname)
    # for format of this file go to UCI ML BoW repo website
    fh = open(fname)
    nd = parse(Int, strip(readline(fh)))
    nw = parse(Int, strip(readline(fh)))
    nnz = parse(Int, strip(readline(fh)))
    D = Array{Int}(nnz)
    W = Array{Int}(nnz)
    WC = Array{Int}(nnz)
    for (idx, ln) in enumerate(eachline(fh))
        D[idx], W[idx], WC[idx] = map(x -> parse(Int, x), split(strip(ln)))
    end
    close(fh)
    # nw x nd sparse matrix
    M = sparse(W, D, WC, nw, nd)
    return M
end

## script
vocab = read_vocab(vf_name)

println("$(length(vocab)) words in vocabulary")
n = 10
println("First $n words in vocabulary")
println(join(vocab[1:30], "\n"))

println("Reading word_doc matrix...")
M = read_word_doc_matrix(dwf_name)
nw, nd = size(M)
println("$nd documents")
