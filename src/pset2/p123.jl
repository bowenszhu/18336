using FillArrays: Fill, Eye
using LinearAlgebra: SymTridiagonal, Diagonal, norm
using SparseArrays: sparse, kron
using Distributions: MvNormal, pdf
using PDMats: ScalMat
using Plots: contour, savefig, plot, plot!
using LaTeXStrings: @L_str
using FFTW: r2r, RODFT00, r2r!, plan_r2r!
using BenchmarkTools: @belapsed
using MAT: matread

const c₀ = 299792458
const μ₀ = π * 4e-7
const ϵ₀ = 8.8541878128e-12

# 1 A fast free-space direct solver
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
     label = "sparse solve", markershape = :circle, legend = :topleft)
plot!(Ns, dst_time, label = "DST precondition", markershape = :circle)
savefig("p14.svg")

# 2 Iterative solver for EM scattering off a head
vars = matread("src/pset2/MRI_DATA.mat")
ϵᵣₑₗ = vars["e_r"]
xs = vec(vars["x"])
ys = vec(vars["y"])
ϵᵣₑₗ_real = real(ϵᵣₑₗ)
ϵᵣₑₗ_imag = imag(ϵᵣₑₗ)
contour(xs, ys, ϵᵣₑₗ_real, aspect_ratio = 1, lims = (0, 1), title = "Re(ϵᵣₑₗ)",
        yflip = true)
savefig("p20_real.svg")
contour(xs, ys, ϵᵣₑₗ_imag, aspect_ratio = 1, lims = (0, 1), title = "Im(ϵᵣₑₗ)",
        yflip = true)
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
f₁ = 21.3e6
ω₁ = 2π * f₁
k² = ω₁^2 * μ₀ * ϵ
∇²k² = fd2 + Diagonal(vec(k²))
u₁ = ∇²k² \ vec(v)
u₁ = reshape(u₁, (N - 1, N - 1))
u₁_crop = copy(u₁)
u₁_crop[crop] .= 0
contour(xs, ys, real(u₁_crop), lim = (0, 1), aspect_ratio = 1, fill = true,
        title = "Re(u), f = $f₁", yflip = true)
savefig("p21_1real.svg")
contour(xs, ys, imag(u₁_crop), lim = (0, 1), aspect_ratio = 1, fill = true,
        title = "Re(u), f = $f₁", yflip = true)
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
        title = "Re(u), f = $f₂", yflip = true)
savefig("p21_2real.svg")
contour(xs, ys, imag(u₂_crop), lim = (0, 1), aspect_ratio = 1, fill = true,
        title = "Re(u), f = $f₂", yflip = true)
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
dst! = plan_r2r!(ū₂, RODFT00)

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
    dst! * u₁
    u₁ ./= factor₁
    dst! * u₁
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
    dst! * u₂
    u₂ ./= factor₂
    dst! * u₂
    u₂ ./= normalize
    error₂[i] = norm(u₂ - ū₂, Inf) / ū₂_norm
end
plot(iteration, error₂, yaxis = :log, xlabel = "iteration",
     ylabel = L"||u^{(l)}-\bar u||_\infty/||\bar u||_\infty", marker = :circle,
     legend = false, title = L"f=%$f₂")
savefig("p22_2.svg")

## Problem 2.3
function iterative_solve(N::Integer, v::AbstractMatrix, f::AbstractFloat)
    u = zeros(ComplexF64, N - 1, N - 1)
    dst! = plan_r2r!(u, RODFT00)
    ω = 2π * f
    k₀² = (ω / c₀)^2
    factor = [k₀² - (λ[i] + λ[j]) / h² for i in 1:(N - 1), j in 1:(N - 1)]
    k²k₀² = ω^2 * μ₀ * ϵ
    k²k₀² .-= k₀²
    normalize = (2N)^2
    for i in 1:19
        @. u = v - k²k₀² * u
        dst! * u
        u ./= factor
        dst! * u
        u ./= normalize
    end
    u
end
iterative_time = @belapsed iterative_solve(256, v, 21.3e6)
iN = only(findall(x -> x == 256, Ns))
sp_time = sp_time[iN]
dst_time = dst_time[iN]
open("p23.txt", "w") do io
    write(io, "iterative_time $iterative_time\n")
    write(io, "sp_time $sp_time\n")
    write(io, "dst_time $dst_time\n")
end

# 3 A spectrally accurate free-space direct solver
## Problem 3.1
N = 256
f = 298.3e6
ω = 2π * f
k² = (ω / c₀)^2
μ = [0.6, 0.7]
σ² = 0.01^2
Σ = ScalMat(2, σ²)
d = MvNormal(μ, Σ)
h = 1 / N
xs = range(h, 1 - h, N - 1)
ys = xs
uv = [pdf(d, [x, y]) for x in xs, y in ys] # v
dst1! = plan_r2r!(uv, RODFT00)
dst1! * uv # v̂
factor = collect(1.0:(N - 1)) # n
factor .^= 2 # n²
factor .*= π^2 # n²π²
factor = factor .+ factor' # n²π² + m²π²
@. factor = k² - factor # k² - n²π² - m²π²
uv ./= factor # û
dst1! * uv
uv ./= (2N)^2 # u
contour(xs, ys, uv, fill = true, aspect_ratio = 1, lims = (0, 1), xlabel = L"x",
        ylabel = L"y", title = L"f=298.3\:\mathrm{MHz}")
savefig("p31.svg")
