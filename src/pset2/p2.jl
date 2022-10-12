using MAT: matread
using Plots: contour, savefig, plot
using FillArrays: Fill, Eye
using Distributions: MvNormal, pdf
using PDMats: ScalMat
using SparseArrays: sparse, kron
using LinearAlgebra: SymTridiagonal, Diagonal, norm
using LaTeXStrings: @L_str
using FFTW: plan_r2r!, RODFT00

# 2 Iterative solver for EM scattering off a head
const μ₀ = π * 4e-7
const ϵ₀ = 8.8541878128e-12
const c₀ = 299792458
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
const N = 256
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
f₁ = 21.3e6
ω₁ = 2π * f₁
k² = ω₁^2 * μ₀ * ϵ
∇²k² = fd2 + Diagonal(vec(k²))
u₁ = ∇²k² \ vec(v)
u₁ = reshape(u₁, (N - 1, N - 1))
u₁_crop = copy(u₁)
u₁_crop[crop] .= 0
contour(xs, ys, real(u₁_crop), lim = (0, 1), aspect_ratio = 1, fill = true,
        title = "Re(u), f = $f₁")
savefig("p21_1real.svg")
contour(xs, ys, imag(u₁_crop), lim = (0, 1), aspect_ratio = 1, fill = true,
        title = "Re(u), f = $f₁")
savefig("p21_1imag.svg")

f₂ = 298.3e6
ω₂ = 2π * f₂
k² = ω₂^2 * μ₀ * ϵ
∇²k² = fd2 + Diagonal(vec(k²))
u₂ = ∇²k² \ vec(v)
u₂ = reshape(u₂, (N - 1, N - 1))
u₂_crop = copy(u₂)
u₂_crop[crop] .= 0
contour(xs, ys, real(u₂_crop), lim = (0, 1), aspect_ratio = 1, fill = true,
        title = "Re(u), f = $f₂")
savefig("p21_2real.svg")
contour(xs, ys, imag(u₂_crop), lim = (0, 1), aspect_ratio = 1, fill = true,
        title = "Re(u), f = $f₂")
savefig("p21_2imag.svg")

## Problem 2.2
ū₁ = u₁
ū₂ = u₂
ū₁_norm = norm(ū₁, Inf)
ū₂_norm = norm(ū₂, Inf)
h = 1 / N
h² = h^2
normalize = (2N)^2
λ = cospi.(range(h, 1 - h, N - 1))
@. λ = 2.0 - 2.0 * λ
iteration = 1:30
plan_dst = plan_r2r!(ū₂, RODFT00)

f₁ = 21.3e6
ω₁ = 2π * f₁
k₀² = (ω₁ / c₀)^2
factor₁ = [k₀² - (λ[i] + λ[j]) / h² for i in 1:(N - 1), j in 1:(N - 1)]
k²k₀² = ω₁^2 * μ₀ * ϵ
k²k₀² .-= k₀²
u₁ = zeros(ComplexF64, N - 1, N - 1)
error₁ = similar(iteration, Float64)
for i in iteration
    @. u₁ = v - k²k₀² * u₁
    plan_dst * u₁
    u₁ ./= factor₁
    plan_dst * u₁
    u₁ ./= normalize
    error₁[i] = norm(u₁ - ū₁, Inf) / ū₁_norm
end
plot(iteration, error₁, yaxis = :log, xlabel = "iteration",
     ylabel = L"||u^{(l)}-\bar u||_\infty/||\bar u||_\infty", marker = :circle,
     legend = false, title = L"f=%$f₁")
savefig("p22_1.svg")

f₂ = 298.3e6
ω₂ = 2π * f₂
k₀² = (ω₂ / c₀)^2
factor₂ = [k₀² - (λ[i] + λ[j]) / h² for i in 1:(N - 1), j in 1:(N - 1)]
k²k₀² = ω₂^2 * μ₀ * ϵ
k²k₀² .-= k₀²
u₂ = zeros(ComplexF64, N - 1, N - 1)
error₂ = similar(iteration, Float64)
for i in iteration
    @. u₂ = v - k²k₀² * u₂
    plan_dst * u₂
    u₂ ./= factor₂
    plan_dst * u₂
    u₂ ./= normalize
    error₂[i] = norm(u₂ - ū₂, Inf) / ū₂_norm
end
plot(iteration, error₂, yaxis = :log, xlabel = "iteration",
     ylabel = L"||u^{(l)}-\bar u||_\infty/||\bar u||_\infty", marker = :circle,
     legend = false, title = L"f=%$f₂")
savefig("p22_2.svg")
