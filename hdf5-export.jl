using ProgressMeter
using ITensors
using HDF5
using Serialization


include("utility.jl")
include("potts-sites.jl")

for dir = abspath.(ARGS)
    @show dir
    fns = [l for l in readdir(dir)
           if !((l == "sites.p")
                || occursin("postprocessed.p", l) 
                || occursin("h5", l)
                )]
    
    sites = deserialize("$dir/sites.p")
    sym = symmetrizer(sites)
    @showprogress for (jθ, fn) in enumerate(fns)
        (θ,E1,E2,Es,ψ) = deserialize("$dir/$fn")
        
        ψsym = symmetrize(sym, ψ)
        d = measure(ψ, sites, [(smb = :X,    tp = Complex{Float64})])
        
        for j in 1:length(ψ)
            h5write("$(dir)/$(fn).h5", "tensor$j", ψ[j] |> array)
        end

        h5write("$(dir)/$(fn).h5", "X", d[:X])
    end
end
