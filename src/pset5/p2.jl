using LinearAlgebra: norm, Diagonal, dot
using TSVD: tsvd
using Statistics: mean

# Problem 2.1
ξ = 6.0
N = M = 100
Y = rand(2, M)
X = rand(2, N)
X[1, :] .+= ξ
G = Matrix{Float64}(undef, M, N)
for n in 1:N
    x = @view X[:, n]
    for m in 1:M
        y = @view Y[:, m]
        G[m, n] = norm(y - x)
    end
end
@. G = log(G)
nvals = 4
U, s, V = tsvd(G, nvals)
G_approx = U * Diagonal(s) * V'
ϕy = vec(sum(G; dims = 2))
ϕy_approx = vec(sum(G_approx; dims = 2))
ϕy_approx .-= ϕy
ϕy_approx ./= ϕy
error = mean(abs, ϕy_approx)

# Problem 2.2
Yₖ = [0.0 1.0 0.0 1.0
      1.0 1.0 0.0 0.0]
Φₖ = Vector{Float64}(undef, 4)
for i in 1:4
    yₖ = @view Yₖ[:, i]
    diff = X .- yₖ
    temp = [norm(@view diff[:, i]) for i in axes(diff, 2)]
    Φₖ[i] = sum(log, temp)
end
weights = Vector{Float64}(undef, 4)
interpolation = Vector{Float64}(undef, M)
for m in 1:M
    x, y = Y[:, m]
    x1 = 1.0 - x
    y1 = 1.0 - y
    weights[1] = x1 * y
    weights[2] = x * y
    weights[3] = x1 * y1
    weights[4] = x * y1
    interpolation[m] = dot(Φₖ, weights)
end
interpolation .-= ϕy
interpolation ./= ϕy
error = mean(abs, interpolation)

# Problem 2.3
Xₖ = copy(Yₖ)
Xₖ[1, :] .+= ξ
dist_temp = similar(X)
potential_corner = Vector{Float64}(undef, 4)
for i in 1:4
    yₖ = @view Yₖ[:, i]
    @. dist_temp = X - yₖ
    temp = [norm(@view dist_temp[:, i]) for i in axes(dist_temp)]
    potential_corner[i] = sum(log, temp)
end
A = [log(norm(view(Yₖ, :, i) - view(Xₖ, :, j))) for i in 1:4, j in 1:4]
Qₖ = A \ potential_corner
collocation = Vector{Float64}(undef, M)
for i in 1:M
    y = @view Y[:, i]
    w = [log(norm(y - @view Xₖ[:, j])) for j in 1:4]
    collocation[i] = dot(w, Qₖ)
end
collocation .-= ϕy
collocation ./= ϕy
error = mean(abs, collocation)
