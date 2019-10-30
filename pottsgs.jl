#julia pottsgs.jl --length 10 --dtheta 0.25 --outdir /home/christopher/work/2019-10-MAGIC/data --subdate TEST

using ITensors
using ArgParse
using ProgressMeter
using Serialization
import ITensors.op

jobname = "M01"

git_commit() = String(read(pipeline(`git log`, `head -1`, `cut -d ' ' -f 2`, `cut -b 1-7`))[1:end-1])
git_commit(path :: String) = cd(git_commit, path)



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

function potts3gs(θ, sites; quiet=false)
    N = length(sites)
    
    if !quiet @show θ end

    ampo = AutoMPO()
    for j = 1:N-1
        add!(ampo, -sin(θ), "Z", j,"ZH",j+1)
        add!(ampo, -sin(θ), "ZH",j,"Z", j+1)
    end
   
    for j = 1:N
        add!(ampo, -cos(θ), "X+XH", j)
    end

    H = toMPO(ampo, sites);
    
    observer = DMRGObserver(Array{String}(undef,0), sites, 1e-7)
    
    sweeps = Sweeps(400)
    maxdim!(sweeps, 10,20,100,100,200)
    cutoff!(sweeps, 1E-10)
    
    ψ0 = randomMPS(sites)
    E1, ψ1 = dmrg(H,ψ0,sweeps, quiet=quiet, observer=observer) 
    
    ψ0= randomMPS(sites)
    E2, ψ2 = dmrg(H,ψ0,sweeps, quiet=quiet, observer=observer)

    if abs(E1 - E2) > 1e-6
        error("Energy difference: θ = $θ, $E1 vs $E2")
    end
    
    #may have gs degeneracy
    #=
    if abs(1 - abs(ovlp)) > 1e-8
        error("Overlap bad: $θ, $ovlp")
    end
    =#
    return E1, ψ1
end

s = ArgParseSettings()
@add_arg_table s begin
    "--length",    "-l" # help = "chain of length"
    "--dtheta"
    "--outdir"
    "--subdate"
end
opts = parse_args(s)

dθ = parse(Float64, opts["dtheta"])
L  = parse(Int64, opts["length"])
outdir  = opts["outdir"]
subdate = opts["subdate"]

itensors_dir = ENV["ITENSORSJL_DIR"]

dir = "$outdir/$subdate/$(git_commit(itensors_dir))-$(git_commit(@__DIR__()))_L$L"
mkpath(dir)

θs = (0.1:dθ:1.9) * π/4
sites = pottsSites(L)
@showprogress for (jθ, θ) in enumerate(θs)
    E, ψ = potts3gs(θ, sites, quiet=true)
    serialize("$(dir)/$(jθ).p", (θ,ψ))
end
