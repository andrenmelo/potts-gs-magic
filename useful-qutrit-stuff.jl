using Test
using TensorOperations

⊗ = kron

d = 3
ω = exp(2*π*im/d)
τ = exp((d + 1)/2 * 2*π*im/3)
@test ω ≈ τ^2

# Clock and shift
X = [0 0 1; 1 0 0; 0 1 0]
Z = Diagonal([1, ω, ω^2])

I3 = [1 0 0; 0 1 0; 0 0 1]
@testset "clock and shift" begin
    @test Z'*Z ≈ I3
    @test X'*X ≈ I3
    @test Z*X ≈ ω * X*Z
end


# Pauli
T(a1 :: Int64 ,a2 :: Int64) = τ^(-a1*a2) * Z^a1 * X^a2

@testset "generalized Pauli operators" begin
    for a1 = 0:2, a2 = 0:2
        @test T(a1,a2)'*T(a1,a2) ≈ I3
        @test T(a1,a2)' ≈ T(3-a1, 3-a2)
        
        for b1 = 0:2, b2 = 0:2
            @test T(b1,b2)*T(a1,a2) ≈ ω^(b1*a2 - b2*a1)*T(a1,a2)*T(b1,b2)
        end
    end
end

# Cliffords 
H = [1 1 1 ;
     1 ω ω^2;
     1 ω^2 ω] * 1/sqrt(3)

F = [1 0 0 ; 
     0 1 0 ;
     0 0 ω ]

S = zeros(Int64, 3,3,3,3)
for m in 0:2, n in 0:2
    S[(n+m)%3 + 1,m+1,n+1,m+1] = 1
end
S = reshape(S, (9,9))

@testset "Clifford gates" begin
    @test (H' * H) ≈ I3 
    @test (F' * F) ≈ I3 
    @test (S' * S) ≈ I3 ⊗ I3 
end

@testset "Clifford action on clock-and-shift" begin
    @test H*Z*H' ≈ X^(3-1)
    @test H*X*H' ≈ Z
    
    @test F*Z*F' ≈ Z
    @test F*X*F' ≈ X*Z
    
    @test S * (Z ⊗ I3) * S' ≈ (Z ⊗ I3) 
    @test S * (X ⊗ I3) * S' ≈ (X ⊗ X)
    
    @test S * (I3 ⊗ Z) * S' ≈ Z^(3-1) ⊗ Z 
    @test S * (I3 ⊗ X) * S' ≈ (I3 ⊗ X)
end

@testset "Clifford action on generalized Paulis" begin
    for a1 = 0:2, a2 = 0:2
        @test H*Z^a1*X^a2*H' ≈ X^(3-a1) * Z^a2
        @test H*T(a1,a2)*H' ≈ T(a2,3-a1)
        
        @test F*Z^a1*X^a2*F' ≈ ω^(-a2*(a2+1)/2) * Z^(a1 + a2) * X^a2
        @test F*T(a1,a2)*F'  ≈ τ^(-a2) * T(a1+a2, a2)
       
        @test S*( (Z^a1*X^a2) ⊗ I3 ) * S' ≈ (Z^a1*X^a2) ⊗ X^a2
        @test S*( I3 ⊗ (Z^a1*X^a2) ) * S' ≈ Z^(3-a1) ⊗ (Z^a1*X^a2) 
        
        for b1 = 0:2, b2 = 0:2
            #=
            @test (S*(      ( Z^a1*X^a2 )      ⊗      ( Z^b1*X^a2 )         ) * S' 
                   ≈   ( Z^(mod(a1-b1,3))*X^a2 ) ⊗ (Z^b1 * X^(mod(a2 + b2, 3))  ) )
            =#
            @test S*( T(a1,a2) ⊗ T(b1,b2) ) * S' ≈ T(a1-b1,a2) ⊗ T(b1, a2+b2)
        end
    end
end


A0 = zeros(Complex{Float64}, (3,3))
for a1 in 0:(d-1), a2 in 0:(d-1)
   global A0 += T(a1, a2)
end 
A0 *= 1/3

function Aμ(a1, a2)
    Ta = T(a1, a2)
    return Ta' * A0 * Ta
end

function ee(j)
    @assert j <= 9
    vct = zeros(9)
    vct[j] = 1
    return vct
end

site_wigner_bchg = zeros(Complex{Float64},9,9)
for (j, (a1,a2)) in enumerate(product(0:2,0:2))
    A = Aμ(a1,a2)
    #A /= norm(A)
    site_wigner_bchg[j,:] = reshape(A', 9)
end

tittums(N2) = reshape(reshape(1:N2, (Int(N2/2),2))', N2) |> collect

function slow_mana(ρ)
    N = length(size(ρ))/2 |> Int
    
    #super inefficient
    W = site_wigner_bchg
    for j in 2:N
        W = W ⊗ site_wigner_bchg
    end
    
    ρ = permutedims(ρ, tittums(2*N))
    ρ = reshape(ρ, length(ρ))
    return 3.0^(-N) * (W * ρ) .|> abs |> sum |> log
end 

function mana(ρ)
    N = length(size(ρ))/2 |> Int
    
    #super inefficient
    W = site_wigner_bchg
   
    ρ = permutedims(ρ, tittums(2*N))
    for j = 1:N
        flush(stdout)
        ρ = reshape(ρ, (cat([9^(j-1), 9, 9^(N-j)], dims=1)...))
        @tensor ρp[α,β,γ] := W[β,βp] * ρ[α, βp, γ]
        ρ = ρp
    end
    return 3.0^(-N) * ρ .|> abs |> sum |> log
end


Nreal = 10
@testset "Faster mana" begin
    @testset "N = 1" begin for r = 1:Nreal
        ψ = rand(Complex{Float64}, 3)
        ψ /= norm(ψ)
        ρ = ψ ⊗ ψ'
        @test abs(mana(ρ) - slow_mana(ρ)) < 1e-14
    end end
    
    @testset "N = 2" begin for r = 1:Nreal
        ψ = rand(Complex{Float64}, 9)
        ψ /= norm(ψ)
        ρ = ψ ⊗ ψ'
        ρ = reshape(ρ, (3,3,3,3))
        @test abs(mana(ρ) - slow_mana(ρ)) < 1e-14
    end end
    
    @testset "N = 3" begin for r = 1:Nreal
        ψ = rand(Complex{Float64}, 27)
        ψ /= norm(ψ)
        ρ = ψ ⊗ ψ'
        ρ = reshape(ρ, (3,3,3,3,3,3))
        @test abs(mana(ρ) - slow_mana(ρ)) < 1e-14
    end end
end

