using ProgressMeter
using ITensors
using HDF5

include("utility.jl")

for dir = abspath.(ARGS)
    @show dir
    fns = [l for l in readdir(dir)
           if !((l == "sites.p") || (occursin("postprocessed.p", l)))]
    
    sym = symmetrizer(sites)
    @showprogress for (jθ, fn) in enumerate(fns)
        (θ,E1,E2,Es,ψ) = deserialize("$dir/$fn")
        
        ψsym = symmetrize(sym, ψ)
        d = measure(ψ, sites, [(smb = :X,    tp = Complex{Float64})])
        
        for j in 1:length(ψ)
            h5write("$(fn).h5", "tensor$j", ψ[j])
        end

        h5write("$(fn).h5", "X", d[:X])
    end
end
