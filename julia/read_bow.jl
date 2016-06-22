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

vocab = read_vocab(vf_name)

println("$(length(vocab)) words in vocabulary")
n = 30
println("First $n words in vocabulary")
println(join(vocab[1:30], "\n"))
