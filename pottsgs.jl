#julia pottsgs.jl --length 10 --dtheta 0.25 --outdir /home/christopher/work/2019-10-MAGIC/data --subdate TEST

using ITensors
using ArgParse
using ProgressMeter
using Serialization
import ITensors.op
import ITensors.randomMPS
include("potts-sites.jl")

git_commit() = String(read(pipeline(`git log`, `head -1`, `cut -d ' ' -f 2`, `cut -b 1-7`))[1:end-1])
git_commit(path :: String) = cd(git_commit, path)

randomMPS(sites, χ :: Int64) = randomMPS(Float64, sites, χ)
function randomMPS(::Type{S}, sites, χ :: Int64) where S <: Number
    N = length(sites)
    links = [Index(χ, "Link,l=$ii") for ii in 1:N-1]

    M = MPS(sites)
    M[1] = randomITensor(S, links[1], sites[1])
    for j = 2:N-1
        M[j] = randomITensor(S, links[j-1], links[j], sites[j])
    end
    M[N] = randomITensor(S, links[N-1], sites[N])
    return M
end

# there has to be a helper function for this
# am too tired
function ket0_mps(sites)
    N = length(sites)
    links = [Index(1, "Link,l=$ii") for ii in 1:N-1]
    M = MPS(sites)

    M[1] = ITensor([0 0 1], links[1], sites[1])
    for j = 2:N-1
        M[j] = ITensor([0 0 1], links[j-1], links[j], sites[j])
    end
    M[N] = ITensor([0 0 1], links[N-1], sites[N])
    return M
end


function potts3gs(θ, λ, χ0, sites;
                  ψ0    :: Union{Nothing,MPS} = nothing,
                  noise :: Bool = false,
                  long  :: Bool = false,
                  quiet :: Bool = false)
    N = length(sites)

    # so I can check the adiabatic deal
    # if ψ0 != nothing @show ψ0[1] |> inds end
    
    if !quiet @show θ end

    ampo = AutoMPO()
    for j = 1:N-1
        add!(ampo, -sin(θ), "Z", j,"ZH",j+1)
        add!(ampo, -sin(θ), "ZH",j,"Z", j+1)
    end
   
    for j = 1:N
        add!(ampo, -cos(θ), "X",  j)
        add!(ampo, -cos(θ), "XH", j)
        add!(ampo, -λ, "Z",  j)
        add!(ampo, -λ, "ZH", j)
    end

    H = MPO(ampo, sites);
    
    if long
        observer = DMRGObserver(Array{String}(undef,0), sites)
    else
        observer = DMRGObserver(Array{String}(undef,0), sites, energy_tol = 1e-7)
    end
    
    sweeps = Sweeps(400)
    maxdim!(sweeps, 10,20,100,100,200)
    cutoff!(sweeps, 1E-7)
    if noise noise!(sweeps, [2.0^(-j) for j in 2:50]...) end

    if ψ0 != nothing
        E1, ψ1 = dmrg(H,ψ0,sweeps, outputlevel=0, observer=observer) 
        E2 = 0
    else #ψ0 == nothing; do it twice
        ψ0 = randomMPS(Complex{Float64}, sites, χ0)
        E1, ψ1 = dmrg(H,ψ0,sweeps, outputlevel=0, observer=observer) 
    
        ψ0= randomMPS(Complex{Float64}, sites, χ0)
        E2, ψ2 = dmrg(H,ψ0,sweeps, outputlevel=0, observer=observer)
        
        if abs(E1 - E2) > 1e-6
            @warn("Energy difference: θ = $θ, $E1 vs $E2")
        end
        #may have gs degeneracy
        #=
        if abs(1 - abs(ovlp)) > 1e-8
        error("Overlap bad: $θ, $ovlp")
        end
        =#
    end

    # so I can check the adiabatic deal
    # @show ψ1[1] |> inds
    # println("-----")
    # flush(stdout)
    
    return E1, E2, observer.energies, ψ1
end

s = ArgParseSettings()
@add_arg_table! s begin
    "--length",    "-l" # help = "chain of length"
    "--dtheta",   default => "0.01"
    "--thetamin", default => "0.1"
    "--thetamax", default => "1.9"
    "--lambda",   default => "0.0"
    "--chi0",     default => "1"
    "--jobname",  default => "M"
    "--noise",    action  => :store_true
    "--adiabatic", action  => :store_true
    "--long",     action  => :store_true
    "--outdir"
    "--subdate"
end
opts = parse_args(s)

dθ = parse(Float64, opts["dtheta"])
λ  = parse(Float64, opts["lambda"])
L  = parse(Int64, opts["length"])
χ0 = parse(Int64, opts["chi0"])
jobname   = opts["jobname"]
noise     = opts["noise"]
adiabatic = opts["adiabatic"]
long      = opts["long"]

θmin = parse(Float64, opts["thetamin"])
θmax = parse(Float64, opts["thetamax"])

outdir  = opts["outdir"]
subdate = opts["subdate"]


using Pkg; itensors_version = Pkg.dependencies()[Pkg.project().dependencies["ITensors"]].version

dir = "$outdir/$jobname/$subdate/$(itensors_version)-$(git_commit(@__DIR__()))_L$L-thetamin$θmin-dtheta$dθ-thetamax$θmax-lambda$λ-chi0$χ0"
if noise     dir = "$dir-noise$noise" end
if adiabatic dir = "$dir-adiabatic$adiabatic" end

mkpath(dir)

θs = (θmin:dθ:θmax) * π/4
if adiabatic θs = cat(θs, reverse(θs[1:end-1]), dims=1) end
@show L, θs, λ
sites = pottsSites(L)
serialize("$(dir)/sites.p", sites)

ψ = randomMPS(sites)

@showprogress for (jθ, θ) in enumerate(θs)
    global ψ
    ψ :: MPS
    if adiabatic
        E1,E2,energies, ψ = potts3gs(θ, λ, χ0, sites, quiet=true, noise=noise, long=long, ψ0 = ψ)
    else
        E1,E2,energies, ψ = potts3gs(θ, λ, χ0, sites, quiet=true, noise=noise, long=long, ψ0 = ket0_mps(sites))
    end
    serialize("$(dir)/$(lpad(jθ,4,'0')).p", (θ,E1,E2,energies,ψ))
    flush(stdout)
end
