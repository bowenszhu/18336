using LinearAlgebra: norm, svdvals
using LaTeXStrings: @L_str
using Plots: plot, savefig

ϵ = 1e-6
N = 20
M = 20
ξs = map(exp10, range(0, 3, 101))
Y = rand(3, M)
X1 = rand(N)
X = rand(3, N)
G = Matrix{Float64}(undef, M, N)
temp = Vector{Float64}(undef, 3)
ranks = zeros(Int, length(ξs))
for (i, ξ) in enumerate(ξs)
    @. X[1, :] = X1 + ξ
    for m in 1:M
        y = @view Y[:, m]
        for n in 1:N
            x = @view X[:, n]
            @. temp = y - x
            G[m, n] = norm(temp)
        end
    end
    @. G = 1.0 / G
    vals = svdvals(G)
    ϵsnorm = ϵ * vals[1]
    for val in vals
        if val < ϵsnorm
            break
        end
        ranks[i] += 1
    end
end
plot(ξs, ranks, axis = :log, legend = false, xlabel = L"ξ", ylabel = "relative rank")
savefig("p1a.svg")

ξ = 6.0
Ns = 20:10:200
ranks = zeros(Int, length(Ns))
for (i, N) in enumerate(Ns)
    local M = N
    local Y = rand(3, M)
    local X = rand(3, N)
    X[1, :] .+= ξ
    local G = Matrix{Float64}(undef, M, N)
    for m in 1:M
        y = @view Y[:, m]
        for n in 1:N
            x = @view X[:, n]
            G[m, n] = norm(y - x)
        end
    end
    @. G = 1.0 / G
    vals = svdvals(G)
    ϵsnorm = ϵ * vals[1]
    for val in vals
        if val < ϵsnorm
            break
        end
        ranks[i] += 1
    end
end
plot(Ns, ranks, legend = false, xlabel = "number of sources/targets",
     ylabel = "relative rank")
savefig("p1b.svg")
