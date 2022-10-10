using FillArrays: Fill, Eye
using LinearAlgebra: SymTridiagonal, Diagonal
using SparseArrays: sparse, kron
using Distributions: MvNormal, pdf
using PDMats: ScalMat
using Plots: contour, savefig, plot, plot!
using LaTeXStrings: @L_str
using FFTW: r2r, RODFT00, r2r!
using BenchmarkTools: @belapsed

# 1 A fast free-space direct solver
const c₀ = 299792458

## Problem 1.1
function ∇²_op(N::Integer)
    h² = 1 / N^2
    ev = Fill(1.0 / h², N - 2)
    dv = Fill(-2.0 / h², N - 1)
    fd1 = sparse(SymTridiagonal(dv, ev))
    eye = Eye{Float64}(N - 1)
    kron(fd1, eye) + kron(eye, fd1)
end
function k²_op(N::Integer, f::AbstractFloat)
    k² = (2π * f / c₀)^2
    Diagonal(Fill(k², (N - 1)^2))
end
function helmholtz_op(N::Integer, f::AbstractFloat)
    fd2 = ∇²_op(N)
    k²_eye = k²_op(N, f)
    fd2 + k²_eye
end
helmholtz_op(8, 21.3e6)

## Problem 1.2
μ = [0.6, 0.7]
σ² = 0.01^2
Σ = ScalMat(2, σ²)
d = MvNormal(μ, Σ) # Gaussian impulse
function v_op(xs::AbstractVector, ys::AbstractVector)
    [pdf(d, [x, y]) for x in xs, y in ys]
end
N = 256
h = 1 / N
xs = range(h, 1 - h, N - 1)
ys = xs
F = v_op(xs, ys)
rhs = vec(F)
fd2 = ∇²_op(N)
f = 21.3e6
helmholtz₁ = fd2 + k²_op(N, f)
u₁ = helmholtz₁ \ rhs
u₁ = reshape(u₁, (N - 1, N - 1))
contour(xs, ys, u₁, fill = true, aspect_ratio = 1, lims = (0, 1), xlabel = L"x",
        ylabel = L"y", title = L"f=21.3\:\mathrm{MHz}")
savefig("p121.svg")
f = 298.3e6
helmholtz₂ = fd2 + k²_op(N, f)
u₂ = helmholtz₂ \ rhs
u₂ = reshape(u₂, (N - 1, N - 1))
contour(xs, ys, u₂, fill = true, aspect_ratio = 1, lims = (0, 1), xlabel = L"x",
        ylabel = L"y", title = L"f=298.3\:\mathrm{MHz}")
savefig("p122.svg")

## Problem 1.3
F̂ = r2r(F, RODFT00)
λ = cospi.(range(h, 1 - h, N - 1))
@. λ = 2.0 - 2.0 * λ
k² = (2π * f / c₀)^2
h² = h^2
for i in 1:(N - 1), j in 1:(N - 1)
    F̂[i, j] /= k² - (λ[i] + λ[j]) / h²
end
U = F̂
r2r!(U, RODFT00)
U ./= (2 * N)^2
contour(xs, ys, U, fill = true, aspect_ratio = 1, lims = (0, 1), xlabel = L"x",
        ylabel = L"y", title = L"f=298.3\:\mathrm{MHz}")
savefig("p13.svg")

## Problem 1.4
function setup_rhs(N::Integer)
    h = 1 / N
    xs = range(h, 1 - h, N - 1)
    ys = xs
    v_op(xs, ys)
end
function sparse_solve(N::Integer, F::AbstractMatrix)
    fd2 = ∇²_op(N)
    helmholtz_op = fd2 + Diagonal(Fill(k², (N - 1)^2))
    U = helmholtz_op \ vec(F)
    reshape(U, (N - 1, N - 1))
end
function dst_precondition(N::Integer, F::AbstractMatrix)
    UF = r2r(F, RODFT00) # F̂
    λ = cospi.(range(h, 1 - h, N - 1))
    @. λ = 2.0 - 2.0 * λ
    h² = 1 / N^2
    for i in 1:(N - 1), j in 1:(N - 1)
        UF[i, j] /= k² - (λ[i] + λ[j]) / h² # Û
    end
    r2r!(UF, RODFT00) # U
    UF ./= (2N)^2 # normalize
    UF
end
Ns = 1 .<< (4:11)
sp_time = similar(Ns, Float64)
dst_time = similar(sp_time)
for (i, N) in enumerate(Ns)
    local F = setup_rhs(N)
    sp_time[i] = @belapsed sparse_solve($N, $F)
    dst_time[i] = @belapsed dst_precondition($N, $F)
end
plot(Ns, sp_time, axis = :log, xlabel = L"N", ylabel = "time [sec]",
     label = "sparse solve", markershape = :circle, legend = :left)
plot!(Ns, dst_time, label = "DST precondition", markershape = :circle)
savefig("p14.svg")
