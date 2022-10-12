using MAT: matread
using Plots: contour, savefig
using FillArrays: Fill, Eye
using Distributions: MvNormal, pdf
using PDMats: ScalMat
using SparseArrays: sparse, kron
using LinearAlgebra: SymTridiagonal, Diagonal
using LaTeXStrings: @L_str

# 2 Iterative solver for EM scattering off a head
const μ₀ = π * 4e-7
const ϵ₀ = 8.8541878128e-12
vars = matread("src/pset2/MRI_DATA.mat")
ϵᵣₑₗ = vars["e_r"]
reverse!(ϵᵣₑₗ; dims = 1)
xs = vec(vars["x"])
ys = vec(vars["y"])
ϵᵣₑₗ_real = real(ϵᵣₑₗ)
ϵᵣₑₗ_imag = imag(ϵᵣₑₗ)
contour(xs, ys, ϵᵣₑₗ_real, aspect_ratio = 1, lims = (0, 1), title = "Re(ϵᵣₑₗ)")
savefig("p20_real.svg")
contour(xs, ys, ϵᵣₑₗ_imag, aspect_ratio = 1, lims = (0, 1), title = "Im(ϵᵣₑₗ)")
savefig("p20_imag.svg")
ϵᵣₑₗ_inner = @view ϵᵣₑₗ[(begin + 1):(end - 1), (begin + 1):(end - 1)]
ϵ = ϵ₀ * ϵᵣₑₗ_inner
xs = @view xs[(begin + 1):(end - 1)]
ys = @view ys[(begin + 1):(end - 1)]

## Problem 2.1
N = 256
μ = [0.5, 0.5]
σ² = 0.01^2
Σ = ScalMat(2, σ²)
d = MvNormal(μ, Σ) # Gaussian impulse
v = [pdf(d, [x, y]) for x in xs, y in ys]
crop = @. abs(ϵᵣₑₗ_inner - 1) < 1e-3
h² = 1 / N^2
ev = Fill(1.0 / h², N - 2)
dv = Fill(-2.0 / h², N - 1)
fd1 = sparse(SymTridiagonal(dv, ev))
eye = Eye{Float64}(N - 1)
fd2 = kron(fd1, eye) + kron(eye, fd1)
f = 21.3e6
ω = 2π * f
k² = ω^2 * μ₀ * ϵ
∇²k² = fd2 + Diagonal(vec(k²))
u = ∇²k² \ vec(v)
u = reshape(u, (N - 1, N - 1))
u[crop] .= 0
contour(xs, ys, real(u), lim = (0, 1), aspect_ratio = 1, fill = true, title = "Re(u), f=$f")
savefig("p21_1real.svg")
contour(xs, ys, imag(u), lim = (0, 1), aspect_ratio = 1, fill = true, title = "Re(u), f=$f")
savefig("p21_1imag.svg")

f = 298.3e6
ω = 2π * f
k² = ω^2 * μ₀ * ϵ
∇²k² = fd2 + Diagonal(vec(k²))
u = ∇²k² \ vec(v)
u = reshape(u, (N - 1, N - 1))
u[crop] .= 0
contour(xs, ys, real(u), lim = (0, 1), aspect_ratio = 1, fill = true, title = "Re(u), f=$f")
savefig("p21_2real.svg")
contour(xs, ys, imag(u), lim = (0, 1), aspect_ratio = 1, fill = true, title = "Re(u), f=$f")
savefig("p21_2imag.svg")
