using ProgressMeter
using ITensors
using Test
using LinearAlgebra
using Serialization
using Statistics
using DataFrames

include("utility.jl")
include("potts-sites.jl")

git_commit() = String(read(pipeline(`git log`, `head -1`, `cut -d ' ' -f 2`, `cut -b 1-7`))[1:end-1])
git_commit(path :: String) = cd(git_commit, path)

postprocess_commit = git_commit(@__DIR__())
itensors_dir = ENV["ITENSORSJL_DIR"]
postprocess_itensor_commit = git_commit(itensors_dir)

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

# wigner function of reduced density matrix on sites jl ... jr (inclusive)
function rdm_wigner(sites, ψ, jl :: Int, jr :: Int)
    orthogonalize!(ψ, jl)
    GC.gc()
    
    if jl <= 1
        Lenv = ITensor(1)
    else
        il = commonindex(ψ[jl-1], ψ[jl])
        Lenv = delta(il,il')
    end

    if jr >= length(sites) || jr == 1 #only going to happen for L = 4
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

    # This sanity check handles certain edge cases (e.g. jr = jl, end of chain) poorly.
    # Not worth the trouble fo fixing atm.
    
    #=
    if(length(inds(Lenv)) + length(inds(Renv)) != 4 + (jr - jl + 1))
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
    =#

    ρ = Lenv * Renv
    
    return ρ
end

function apply_wigner_bchg(sites, ρ :: ITensor)
    for j in 1:length(sites)
        flush(stdout)
        ρ    *=  wigner_bchg(sites[j],3)* (1/3)
    end
    return ρ 
end

# mana of a reduced density matrix (as a single ITensor)
# assumes already in wigner basis
function mana(ρ :: ITensor)
    W = array(ρ)
    @assert abs(sum(W) - 1) ≤ 1e-9
    return W .|> abs |> sum |> log
end

# mana of rdm of state ψ on sites jl ... jr inclusive
function mana(sites, ψ :: MPS, jl :: Int, jr :: Int)
    W = rdm_wigner(sites, ψ, jl, jr)
    flush(stdout)
    m = mana(W)
    return m
end

# site arithmetic: length-l subset of length-N chain
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
    #@showinds Lenv
    for j = j1:j2
        Lenv *= ψ[j]
        Lenv *= dag(prime(ψ[j], "Link"))
        #@showinds Lenv
    end
    return Lenv
end


# reduced density matrix of state ψ on union of two widely separated regions
#     ψ:     state
#     sites: ITensor site objects for ψ
#     jl:    left point
#     jrs:   list of right points
#     x:     size of region
# e.g.
#
#     twopoint_rdm(ψ, sites, 3, 10:12, 2)
#
# returns rdms on
#
#     [ 3 10 11 ]
#     [ 3 11 12 ]
#     [ 3 12 13 ]
#
# Note that this explicitly constructs all of the rdms; for x >~ 2,
# this is Bad. Better would be some kind of iterator.

twopoint_rdm(ψ :: MPS, sites, jl :: Int, jrs) = twopoint_rdm(ψ :: MPS, sites, jl :: Int, jrs, 1)
function twopoint_rdm(ψ :: MPS, sites, jl :: Int, jrs, x :: Int)

    jrs = sort(jrs)
    ρs         = Array{ITensor}(undef, length(jrs))
    out_sites  = Array{Any}(undef, length(jrs))
    N = length(sites)
    
    orthogonalize!(ψ, jl)
   
    #---
    # set up the left environment
    if jl <= 1
        Lenv = ITensor(1)
    else
        il = setdiff(findinds(ψ[jl], "Link"), commoninds(ψ[jl], ψ[jl+1]))[1]
        Lenv = delta(il,il')
    end

    # left side: construct x-site rdm starting at jl,
    # with dangling virtual index
    lsites = jl:min(N, jl + x - 1)
    @show x
    println("L part")
    for j = lsites
        @show j
        Lenv *= ψ[j]
        Lenv *= ψ[j] |> prime |> dag
    end

    
    #---
    # work through the jrs, pulling out an rdm at each
    jmarker = jl+x
    ctr = 1
    for jr in jrs

        # If jr is in the region we already did, error out
        if jr <= jl + x - 1 error("jr <= jl + x - 1: jr = $jr, jl = $jr, x = $x") end 

        Lenv = propagate(Lenv, ψ, jmarker, jr-1)

        jrightmost = min(N, jr + x - 1)
        if jrightmost >= length(sites)
            Renv = ITensor(1)
        else
            ir = setdiff(findinds(ψ[jrightmost], "Link"), commoninds(ψ[jrightmost-1], ψ[jrightmost]))[1]
            Renv = delta(ir,ir')
        end

        
        # right side: construct x-site rdm ending at jr,
        # with dangling virtual index

        rsites = jrightmost :-1:jr
        println("R part")
        for j = rsites
            @show j
            Renv *= ψ[j]
            Renv *= ψ[j] |> prime |> dag
        end
        
        ρs[ctr] = Lenv * Renv
        out_sites[ctr] = sites[lsites ∪ rsites]
        
        ctr += 1
        jmarker = jr

    end

    
    return(ρs, out_sites)
end


function ZZH(sites, ρ)
    @assert 2 == length(sites)
    @assert 2 * length(sites) == ρ |> inds |> length
    ZZHop = op(sites[1],"Z") * op(sites[2],"ZH") 
    ZIdop = op(sites[1],"Z") * delta(sites[2], sites[2]')
    IdZop = delta(sites[1], sites[1]') * op(sites[2],"ZH")
    return ITensors.scalar( (ρ * ZZHop) - (ρ*ZIdop)*(ρ*IdZop) )
end 

# misnomer
# really not restricted to twopoint situation
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
    # would be better to affirmatively specify files to read as regex
    fns = [l for l in readdir(dir)
           if !((l == "sites.p") || (occursin("postprocessed.p", l)))]
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
                    :stpmn  => Array{Array{Float64,2}}(undef, Nθ),
                    :slmn  => Array{Array{Float64,1}}(undef, Nθ),
                    :srmn  => Array{Array{Float64,2}}(undef, Nθ),
                    :stpS2  => Array{Array{Float64,1}}(undef, Nθ),
                    :ZZH    => Array{Array{Complex{Float64},1}}(undef, Nθ)] )

    qs = [(smb = :X,    tp = Complex{Float64}),
          (smb = :XH,   tp = Complex{Float64}),
          (smb = :Z,    tp = Complex{Float64}),
          (smb = :ZH,   tp = Complex{Float64})]

    sites = deserialize("$dir/sites.p")
    L = length(sites)
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

        orthogonalize!(ψ, L)
        ψ[L] /= ψ[L] |> array |> norm

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


        # two-point mana for region sizes xs = 1:3
        # doing this as an array is kinda weird
        xs = 1:2
        stpmn = zeros(3*L/4 |> Int, length(xs)) 
        for x in xs
            jrs = (jl + x) : L
            ρs, tpsites = twopoint_rdm(ψsym, sites, jl, jrs, x)
            stpmn[x:end,x] = [twopoint_mana(sts, ρ) for (ρ, sts) in zip(ρs, tpsites)]
            if x == 1
                df[jθ, :ZZH]   = [ZZH(sts, ρ)           for (ρ, sts) in zip(ρs, tpsites)]
                df[jθ, :stpS2] = [S2(ρ)                 for (ρ, sts) in zip(ρs, tpsites)]
            end
        end
        @show stpmn[:,2]

        # twopoint_rdm puts the oc at jl.
        # We need the manas of the rdms of the two regions separately;
        #
        #     mana(sites, ψ :: MPS, jl :: Int, jr :: Int)
        #
        # calls
        #
        #     rdm_wigner(sites, ψ, jl :: Int, jr :: Int)
        # 
        # which puts the oc at *that* jl i.e. the left edge of the
        # subsystem whose mana we want.
        #
        # The arrangement I've got here, then, walks the o.c. through the chain.
        
        slmn = [mana(sites, ψ, jl, jl + x - 1) for x in xs]
        srmn  = zeros( 3*L/4 |> Int, length(xs))

        # order of for loops will be super important for performance
        for (jjr, jr) in enumerate(Int(L/4+1):L) # this idiom's a little funny
            for (jx,x) in enumerate(xs)
                jrightmost = min(L, jr + x - 1)
                if jrightmost > L/4 + x - 1
                    srmn[jjr, jx] = mana(sites, ψ, jr, jrightmost)
                end
            end
        end
        @show srmn[:,2]
        
        df[jθ, :stpmn] = stpmn
        df[jθ, :slmn] = slmn
        df[jθ, :srmn] = srmn
        df[jθ, :θ]     = θ
        df[jθ, :jl]    = jl
        
        GC.gc()
    end

    mn_df = DataFrame()
    mn_df.θ = arr1d(θs)
    mn_df.L = L
    for (jl,l) in enumerate(ls)
        mn_df[!,Symbol("mn$l")]  .= mn[:,jl]
        mn_df[!,Symbol("smn$l")] .= sym_mn[:,jl]
    end

    fn = "$dir/$(postprocess_commit)_postprocessed.p"
    serialize(fn, (df, mn_df, postprocess_commit, postprocess_itensor_commit))
    f = open(ENV["MAGIC_POSTPROCESSED"], "a")
    println(f, fn)
    close(f)
end
