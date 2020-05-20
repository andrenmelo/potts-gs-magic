function symmetrizer(sites)
    N = length(sites)
    I  = MPO(sites, Vector{String}(["Id"   for j in 1:N]))
    X  = MPO(sites, Vector{String}(["X"    for j in 1:N]))
    X2 = MPO(sites, Vector{String}(["XH" for j in 1:N]))
    sym = sum(I,X)
    sym = sum(sym, X2)
    sym[1] *= (1.0 / sqrt(3) )
    return sym
end

#builtin is somehow pathological: out of memory error
function cdw_applyMPO(A :: MPO, ψ :: MPS)
    φA = [mapprime(ψ[j] * A[j], 1, 0) for j in 1:length(ψ)]
    φ = MPS(length(ψ), φA)    
    return φ
end

function symmetrize(sym, ψ :: MPS)
    N = length(ψ)
    ψsym = cdw_applyMPO(sym, ψ)
    orthogonalize!(ψsym, 1)
    
    ψsym[N] /= sqrt(inner(ψsym, ψsym))#setindex! sets llim and rlim appropriately
    orthogonalize!(ψsym, 1)
    return ψsym
end

function measure(M :: MPS, sites :: Array{Index,1}, qs)
    N = length(M)
    @assert length(sites) == N
    orthogonalize!(M, N)

    d = Dict([q.smb => zeros(q.tp, N) for q in qs])
    d = convert(Dict{Symbol, Any}, d)
    #bond labeled by site to its left
    d[:s] = Array{Array{Float64,1}}(undef, N-1)

    for j in reverse(2:N)
        rinds = uniqueinds(M[j],M[j-1])

        #at this point the orthogonality center lives on site j
        A = M[j] 
        Adag = prime(A, "Site") |> dag
        [d[q.smb][j] = A * op(sites[j], string(q.smb)) * Adag |> scalar for q in qs]

        U,S,V = svd(M[j],rinds)
        M[j] = U
        M[j-1] *= (S*V)
        ITensors.setRightLim!(M,j)

        # measurements
        s = S |> array
        d[:s][j-1] = s[CartesianIndex.(axes(s)...)]
    end
    d[:χ] = [length(s) for s in d[:s]]
    d[:SvN] = [-sum(s .* lg.(s)) for s in d[:s]]
    return d
end

