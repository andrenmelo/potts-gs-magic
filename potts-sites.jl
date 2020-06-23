import ITensors.op

function pottsSites(N :: Int; q :: Int = 3)
  return [Index(q, "Site,Potts,n=$n") for n = 1:N]
end
const PottsSite = TagType"Potts"

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
