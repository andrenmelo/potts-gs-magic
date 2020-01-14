using ProgressMeter
using ITensors
using Test
using LinearAlgebra
using Serialization
using Statistics
using DataFrames


git_commit() = String(read(pipeline(`git log`, `head -1`, `cut -d ' ' -f 2`, `cut -b 1-7`))[1:end-1])
git_commit(path :: String) = cd(git_commit, path)

postprocess_commit = git_commit(@__DIR__())
itensors_dir = ENV["ITENSORSJL_DIR"]
postprocess_itensor_commit = git_commit(itensors_dir)

lg(x) = log(x)/log(2)
arr1d(a :: Array) = reshape(a, length(a))

function vectorspace(Lst :: Array{T}) where T
    sz = size(Lst[1])
    lngth = length(Lst[1])
    N = length(Lst)
    
    @assert [sz == size(A) for A in Lst] |> all
    
    lst = [reshape(A, lngth) for A in Lst]
    C = cat(lst..., dims=1)
    reshape(C, (sz..., N))
    return C
end

vectorspace(Lst :: Array{<:ITensor}) = vectorspace(Lst, "Vspc" )
function vectorspace(Lst :: Array{<:ITensor}, tags... ) :: ITensor
    is = inds(Lst[1])
    @assert([inds(A) == is for A in Lst] |> all)
    V = vectorspace(array.(Lst))
   
    inew = Index(length(Lst))
    for t in tags
        inew = addtags(inew, t)
    end
    ITensor(V, is..., inew)
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

import ITensors.op

function pottsSites(N :: Int; q :: Int = 3)
  return [Index(q, "Site,Potts,n=$n") for n = 1:N]
end

const PottsSite = makeTagType("Potts")

# 1-index my Potts states
# so diagonal elements of Z are
#    e^{2πi/q}
#    e^{2πi*2/q}
#    e^{2πi*3/q}
#    e^{2πi*(q-1)/q}
#    e^{2πi*q/q} = 1
# this seems like the least bad thing
function state(::PottsSite,
               st::AbstractString)
  return parse(Int64, st)
end

function op(::PottsSite,
            s :: Index,
            opname :: AbstractString)::ITensor
  sP = prime(s)
  q = dim(s)

  Op = ITensor(Complex{Float64},dag(s), s')

  if opname == "Z"
    for j in 1:q
      Op[j,j] = exp(2*π*im*j/q)
    end
  elseif opname == "ZH"
    for j in 1:q
      Op[j,j] = exp(-2*π*im*j/q)
    end
  elseif opname == "X"
    for j in 1:q
      Op[(j % q) + 1,j] = 1
    end
  elseif opname == "XH"
    for j in 1:q
      Op[j,(j % q) + 1] = 1
    end
  elseif opname == "X+XH"
    for j in 1:q
      Op[j,(j % q) + 1] = 1
      Op[(j % q) + 1,j] = 1
    end
  else
    throw(ArgumentError("Operator name '$opname' not recognized for PottsSite"))
  end
  return Op
end    


q = 3
X = zeros(q,q)
for j in 1:q
    X[(j % q) + 1, j] = 1
end
Z = Diagonal([exp(2*π*im*j/q) for j in 1:q])
ω = exp(2*π*im/q)
T(a1, a2) = (exp((4)*π*im/3))^(-a1*a2) * Z^a1 * X^a2
A0 = zeros(Complex{Float64}, (3,3))
for a1 in 1:q, a2 in 1:q
   global A0 += T(a1, a2)
end 
A0 *= 1/3

function Aμ(a1, a2)
    Ta = T(a1, a2)
    return Ta' * A0 * Ta
end

wigner(a1, a2, i :: Index) =  ITensor(Aμ(a1,a2), i, i')

# this is a disaster
# not even close to encapsulated
function wigner_bchg(i :: Index, q)
    wigners =  [wigner(a1,a2,i) for a1 in 0:q-1, a2 in 0:q-1]
    return vectorspace(reshape(wigners, length(wigners)), "Site", "Vspc")
end



function rdm_wigner(sites, ψ, jl :: Int, jr :: Int)
    orthogonalize!(ψ, jl)
    GC.gc()
    
    if jl <= 1
        Lenv = ITensor(1)
    else
        il = setdiff(findinds(ψ[jl], "Link"), commoninds(ψ[jl], ψ[jl+1]))[1]
        Lenv = delta(il,il')
    end

    if jr >= length(sites)
        Renv = ITensor(1)
    else
        ir = setdiff(findinds(ψ[jr], "Link"), commoninds(ψ[jr], ψ[jr-1]))[1]
        Renv = delta(ir,ir')
    end
    
    jmid = (jl + jr)/2 |> floor |> Int
   
    for j = jl:jmid
        Lenv *=  ψ[j]
        Lenv *= (ψ[j] |> prime |> dag) * (wigner_bchg(sites[j],3) * (1/3) )
    end
    for j = jr:-1:(jmid + 1)
        Renv *=  ψ[j]
        Renv *= (ψ[j] |> prime |> dag) * (wigner_bchg(sites[j],3) * (1/3) )
    end

    if(length(inds(Lenv)) + length(inds(Renv)) != 4 + 2*(jr - jl + 1))
        @show jl, jr
	flush(stdout)
        println("Lenv")
        Lenv |> inds .|> println
	flush(stdout)
        println("Renv")
        Renv |> inds .|> println
	flush(stdout)
	error()
    end

    #@show eltype(Lenv)
    #@show eltype(Renv)
    ρ = Lenv * Renv
    
    return ρ
end

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

function apply_wigner_bchg(sites, ρ :: ITensor)
    for j in 1:length(sites)
        flush(stdout)
        ρ    *=  wigner_bchg(sites[j],3)* (1/3)
    end
    return ρ 
end

#assumes already in wigner basis
function mana(ρ :: ITensor)
    W = array(ρ)
    @assert abs(sum(W) - 1) ≤ 1e-9
    return W .|> abs |> sum |> log
end

function mana(sites, ψ :: MPS, jl :: Int, jr :: Int)
    W = rdm_wigner(sites, ψ, jl, jr)
    flush(stdout)
    m = mana(W)
    return m
end
function middlesection(N, l)
    j = N/2 |> floor |> Int
    jl = j - (l/2 |> floor |> Int)
    jr = j + (l/2 |> ceil |> Int) - 1

    if jl < 1 jl = 1 end
    if jr > N jr = N end
    
    return (jl, jr)
end


function unitvector(ind :: Index, j :: Int)
    e = ITensor(0, ind)
    e[ind[j]] = 1
    return e
end
ind = Index(3)

import ITensors

MPS(v :: Vector{<:ITensor}) = MPS(length(v), v)
MPO(v :: Vector{<:ITensor}) = MPO(length(v), v)


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


#Multiply Lenv by transfer matrices on sites j1 ... j2 inclusive
function propagate(Lenv, ψ, j1, j2)
    #println("enter propogate")
    #@showinds Lenv
    for j = j1:j2
        Lenv *= ψ[j]
        Lenv *= dag(prime(ψ[j], "Link"))
        #@showinds Lenv
    end
    return Lenv
end

function twopoint_rdm(ψ :: MPS, sites, jl :: Int, jrs)
    
    jrs = sort(jrs)
    ρs  = Array{ITensor}(undef, length(jrs))
    
    orthogonalize!(ψ, jl)
   
    #---
    # set up the left environment
    if jl <= 1
        Lenv = ITensor(1)
    else
        il = setdiff(findinds(ψ[jl], "Link"), commoninds(ψ[jl], ψ[jl+1]))[1]
        Lenv = delta(il,il')
    end

    Lenv *= ψ[jl]
    Lenv *= ψ[jl] |> prime |> dag
    #---
    # work through the jrs, pulling out an rdm at each
    jmarker = jl+1
    ctr = 1
    for jr in jrs
        Lenv = propagate(Lenv, ψ, jmarker, jr-1)
        
        if jr >= length(sites)
            Renv = ITensor(1)
        else
            ir = setdiff(findinds(ψ[jr], "Link"), commoninds(ψ[jr], ψ[jr-1]))[1]
            Renv = delta(ir,ir')
        end
        
        Lenvp = (Lenv * ψ[jr]) * dag(prime(ψ[jr]))
        #=
        @showinds Lenv
        ψjr = ψ[jr]
        @showinds ψjr
        @showinds Lenvp
        @showinds Renv
        =#
        ρs[ctr] = Lenvp * Renv
        
        ctr += 1
        jmarker = jr
    end

    return(ρs, [sites[[jl, jr]] for jr in jrs])
end


function ZZH(sites, ρ)
    @assert 2 == length(sites)
    @assert 2 * length(sites) == ρ |> inds |> length
    ZZHop = op(sites[1],"Z") * op(sites[2],"ZH") 
    ZIdop = op(sites[1],"Z") * delta(sites[2], sites[2]')
    IdZop = delta(sites[1], sites[1]') * op(sites[2],"ZH")
    return ITensors.scalar( (ρ * ZZHop) - (ρ*ZIdop)*(ρ*IdZop) )
end 

function twopoint_mana(sites, ρ)
    ρ = apply_wigner_bchg(sites, ρ)
    return mana(ρ)
end

function S2(ρ :: ITensor)
    return -2*lg(norm(ρ))
end

for dir = abspath.(ARGS)
    @show dir

    ls = (1:7) |> reverse

    #fns = readdir(dir)[1:end-1]
    fns = [l for l in readdir(dir) if !(l ∈ ["postprocessed.p", "sites.p"]) ]
    Nθ = length(fns)
    θs = Array{Float64}(undef, Nθ)

    trueNθ = Nθ



    
    df = DataFrame([:θ     => Array{Float64}(undef, Nθ),
                    :L     => Array{Int64}(undef, Nθ),
                    :jl    => Array{Int64}(undef, Nθ),
                    :jrs   => Array{Array{Int64,1}}(undef, Nθ),
                    :direction => Array{Symbol}(undef, Nθ),
                    :E2 => Array{Float64}(undef, Nθ),
                    :E1 => Array{Float64}(undef, Nθ),
                    :SvNmax => Array{Float64}(undef, Nθ),
                    :measX  => Array{Complex{Float64}}(undef, Nθ),
                    :measZ  => Array{Complex{Float64}}(undef, Nθ),
                    :chimax => Array{Int64}(undef, Nθ),
                    :stpmn  => Array{Array{Float64,1}}(undef, Nθ),
                    :stpS2  => Array{Array{Float64,1}}(undef, Nθ),
                    :ZZH    => Array{Array{Complex{Float64},1}}(undef, Nθ)] )

    qs = [(smb = :X,    tp = Complex{Float64}),
          (smb = :XH,   tp = Complex{Float64}),
          (smb = :Z,    tp = Complex{Float64}),
          (smb = :ZH,   tp = Complex{Float64})]

    sites = deserialize("$dir/sites.p")
    sym = symmetrizer(sites)
    mn     = Array{Float64}(undef, (Nθ, length(ls)))
    sym_mn = Array{Float64}(undef, (Nθ, length(ls)))
    @showprogress for (jθ, fn) in enumerate(fns)
        if jθ < (trueNθ/2 + 1)
            df[jθ, :direction] = :fromdisordered
        else
            df[jθ, :direction] = :fromordered
        end
       
        (θ,E1,E2,Es,ψ) = deserialize("$dir/$fn")
	θs[jθ] = θ
        L = length(sites)
        d = measure(ψ, sites, qs)
        
        df[jθ, :L]     = L
        df[jθ, :θ]     = θ
        df[jθ, :measX] = mean(d[:X]) + mean(d[:XH])
        df[jθ, :measZ] = mean(d[:Z]) + mean(d[:ZH])
        df[jθ, :E1]    = E1
        df[jθ, :E2]    = E2
        df[jθ, :SvNmax] = maximum(d[:SvN])
        df[jθ, :chimax] = maximum(d[:χ])
        
        flush(stdout)
        ψsym = symmetrize(sym, ψ)
        for (jl, l) in (ls |> enumerate)
            mn[jθ,jl]     = mana(sites, ψ,    middlesection(L,l)...)
            sym_mn[jθ,jl] = mana(sites, ψsym, middlesection(L,l)...)
        end


        #bad hygiene: re-use of this variable jl
        jl = L/4 |> Int
        jrs = (jl + 1) : L

        
        ρs, tpsites = twopoint_rdm(ψsym, sites, jl, jrs)
        df[jθ, :θ]     = θ
        df[jθ, :stpmn] = [twopoint_mana(sts, ρ) for (ρ, sts) in zip(ρs, tpsites)]
        df[jθ, :ZZH]   = [ZZH(sts, ρ)           for (ρ, sts) in zip(ρs, tpsites)]
        df[jθ, :stpS2] = [S2(ρ)                 for (ρ, sts) in zip(ρs, tpsites)]
        df[jθ, :jl]    = jl
        df[jθ, :jrs]   = jrs
        
        GC.gc()
    end

    mn_df = DataFrame([[:θ=>arr1d(θs)];
                       [Symbol("mn$l")  =>     mn[:,jl] for (jl, l) in enumerate(ls)];
                       [Symbol("smn$l") => sym_mn[:,jl] for (jl, l) in enumerate(ls)]
                       ])

    fn = "$dir/postprocessed.p"
    serialize(fn, (df, mn_df, postprocess_commit, postprocess_itensor_commit))
    f = open(ENV["MAGIC_POSTPROCESSED"], "a")
    println(f, fn)
    close(f)
end
