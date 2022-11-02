using LinearAlgebra: mul!, Diagonal, norm
using Plots: plot, savefig
using LaTeXStrings: @L_str

# Problem 2.2
const N = 64
const n = 3
θ = 0:(1 / (N - 1)):1
x = cospi.(θ)
@. x = 0.5 * (x + 1)
D = Matrix{Float64}(undef, N, N)
temp = (1 + 2 * (N - 1)^2) / 6
D[begin] = temp
D[end] = -temp
for j in 2:(N - 1)
    D[j, j] = -x[j] / (2 * (1 - x[j]^2))
end
c = ones(N)
c[begin] += 1
c[end] += 1
for i in 1:N
    for j in 1:N
        if i == j
            continue
        end
        D[i, j] = c[i] / (c[j] * (x[i] - x[j])) * (-1)^(i + j)
    end
end
u = @. 5 * (1 - x)
xD²2D = x .* D * D + 2 * D
rhs = similar(u)
A = similar(xD²2D)
norm_v = Float64[]
while true
    @. rhs = x * u^n
    mul!(rhs, xD²2D, u, true, true)
    rhs[begin] = 0
    A .= xD²2D
    A .+= Diagonal(@. n * x * u^(n - 1))
    A[1, 1] = 1
    A[1, 2:end] .= 0
    v = A \ rhs
    u .-= v
    push!(norm_v, norm(v, Inf))
    if norm_v[end] < 1e-10
        break
    end
end
plot(norm_v, yaxis = :log10, legend = false, xlabel = L"n",
     ylabel = L"|v^n|_\infty")
savefig("p22.svg")

# Problem 2.3
