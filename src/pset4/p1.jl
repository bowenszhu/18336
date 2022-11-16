using FillArrays: Fill
using SparseArrays: spdiagm
using LinearAlgebra: Diagonal, lu
using Plots: heatmap, plot, plot!, savefig
using BenchmarkTools: @belapsed
using LaTeXStrings: @L_str
using FFTW: plan_r2r!, REDFT00

const L = 300.0
const b = 0.5
const c = -1.76

# Problem 1.2
function S⁰(N::Integer)
    sd = Fill(-0.5, N - 2)
    d = Fill(0.5, N)
    res = spdiagm(0 => d, 2 => sd)
    res[1, 1] = 1.0
    res
end
function S¹(N::Integer)
    d = 1.0 ./ (1.0:N)
    sd = d[3:end]
    @. sd = -sd
    spdiagm(0 => d, 2 => sd)
end
function ultraspherical(N::Integer, h)
    D² = spdiagm(2 => 2.0 * (2.0 / L)^2 .* (2.0:(N - 1)))
    Mₐ⁰ = Diagonal(Fill(1.0 / h - 1.0, N))
    Mₐ² = Diagonal(Fill(-1.0 - b * im, N))
    M = S¹(N) * S⁰(N) * Mₐ⁰ + Mₐ² * D²
    M[end, :] .= 1.0
    M[end - 1, 1:2:end] .= 1.0
    M[end - 1, 2:2:end] .= -1.0
end

N = 32
h = 0.02
lhs_op = ultraspherical(N, h)
struc = log.(Matrix(abs.(lhs_op)))
heatmap(struc, yflip = true, aspect_ratio = 1, lims = (0, N + 1))

# Problem 1.3
Ns = 1 .<< (7:13)
time = similar(Ns, Float64)
for (i, N) in enumerate(Ns)
    LHS = ultraspherical(N, h)
    LU = lu(LHS)
    F = rand(N)
    time[i] = @belapsed $LU \ $F
end
plot(Ns, time, axis = :log, xlabel = L"N", ylabel = "time [sec]", legend = false,
     marker = :circle)

N = 32
F = rand(N)
LHS = ultraspherical(N, h)
LU = lu(LHS)
res1 = LU \ F
LU2 = lu(LHS')
res2 = LU2' \ F
res1 ≈ res2

time_woodbury = similar(time)
for (i, N) in enumerate(Ns)
    LHS = ultraspherical(N, h)
    LU = lu(LHS')
    F = rand(N)
    time_woodbury[i] = @belapsed $(LU') \ $F
end
plot(Ns, time, axis = :log, xlabel = L"N", ylabel = "time [sec]", marker = :circle,
     label = "LU", legend = :topleft)
plot!(Ns, time_woodbury, marker = :circle, label = "transpose-based")

# Problem 1.4
function rhs_op(A, S1S0, dct1!)
    N = length(A)
    rhs = copy(A)
    dct1! * rhs
    @. rhs = rhs / h - (1.0 + c * im) * abs2(rhs) * rhs
    dct1! * rhs
    rhs ./= 2(N - 1)
    rhs = S1S0 * rhs
    rhs[end - 1, :] .= 0
    rhs
end
time_rhs = similar(time)
for (i, N) in enumerate(Ns)
    A = rand(ComplexF64, N)
    S1S0 = S¹(N) * S⁰(N)
    dct1! = plan_r2r!(A, REDFT00)
    time_rhs[i] = @belapsed rhs_op($A, $S1S0, $dct1!)
end
plot(Ns, time, axis = :log, xlabel = L"N", ylabel = "time [sec]", marker = :circle,
     label = "LU", legend = :topleft)
plot!(Ns, time_woodbury, marker = :circle, label = "transpose-based")
plot!(Ns, time_rhs, marker = :circle, label = "RHS")

# Problem 1.5
N = 1024
x = cospi.(range(0, 1, N))
x .+= 1.0
A = @. ComplexF64(1e-3 * sinpi(x))
x .*= L / 2
S1S0 = S¹(N) * S⁰(N)
LHS = ultraspherical(N, h)
LU = lu(LHS')
dct1! = plan_r2r!(A, REDFT00)
amplitude = Matrix{Float64}(undef, N, 3000)
phase = similar(amplitude)
for j in 1:3000
    for i in 1:10
        RHS = rhs_op(A, S1S0, dct1!)
        A = LU' \ RHS
    end
    @. amplitude[:, j] = abs2(A)
    @. phase[:, j] = atan(real(A), imag(A))
end
plot(amplitude, legend = false, title = "amplitude", xlabel = L"x")
plot(x, phase, legend = false, title = "phase", xlabel = L"x")
