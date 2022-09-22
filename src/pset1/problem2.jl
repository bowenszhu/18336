using FFTW, LinearAlgebra, BenchmarkTools, Plots

function DFT_matrix(N::Integer)
    ns = collect(Float64, 0:(N - 1))
    m = ns * ns'
    m .*= -2 / N
    cispi.(m)
end

function apply_DFT(x::AbstractMatrix)
    N = size(x, 1)
    DFT_matrix(N) * x
end

N = 512
M = 1
x = rand(ComplexF64, N, M)
mine = apply_DFT(x)
fftw = fft(x)
diff = mine - fftw
norm(diff)

Ns = 1 .<< (1:10)
DFT_time = similar(Ns, Float64)
FFT_time = similar(DFT_time)
M = 1
for (i, N) in enumerate(Ns)
    println(i, " ", N)
    local x = rand(ComplexF64, N, M)
    DFT_time[i] = @belapsed apply_DFT($x)
    FFT_time[i] = @belapsed fft($x)
end
plt_m1 = plot(Ns, DFT_time, label = "DFT", xlabel = "N", ylabel = "time [sec]",
              axis = :log10, minorgrid = true, markershape = :circle, title = "M = $M",
              legend = :topleft)
plot!(plt_m1, Ns, FFT_time, label = "FFT", markershape = :rect)

DFT_cache = map(DFT_matrix, Ns)

function apply_cached_DFT(x::AbstractMatrix)
    N = size(x, 1)
    # assume N is a power of 2
    i = trailing_zeros(N)
    DFT_cache[i] * x
end

cached_DFT_time = similar(DFT_time)
for (i, N) in enumerate(Ns)
    println(i, " ", N)
    local x = rand(ComplexF64, N, M)
    cached_DFT_time[i] = @belapsed apply_cached_DFT($x)
end
plot!(plt_m1, Ns, cached_DFT_time, label = "cached DFT", markershape = :star5)

M = 100
Ns = 1 .<< (1:10)
DFT_time = similar(Ns, Float64)
cached_DFT_time = similar(DFT_time)
FFT_time = similar(DFT_time)
for (i, N) in enumerate(Ns)
    println(i, " ", N)
    local x = rand(ComplexF64, N, M)
    DFT_time[i] = @belapsed apply_DFT($x)
    cached_DFT_time[i] = @belapsed apply_cached_DFT($x)
    FFT_time[i] = @belapsed fft($x)
end
plt_m100 = plot(Ns, DFT_time, label = "DFT", xlabel = "N", ylabel = "time [sec]",
                axis = :log10, minorgrid = true, markershape = :circle, title = "M = $M",
                legend = :topleft)
plot!(plt_m100, Ns, FFT_time, label = "FFT", markershape = :rect)
plot!(plt_m100, Ns, cached_DFT_time, label = "cached DFT", markershape = :star5)

function apply_FFT(x::AbstractMatrix)
    x = convert.(ComplexF64, x)
    N = size(x, 1)
    if N == 1
        return x
    end
    y = similar(x)
    ωₙ = cispi.(range(0; step = -2 / N, length = N >> 1))
    apply_FFT!(y, x, ωₙ, N)
    return y
end
function apply_FFT!(y::AbstractMatrix{ComplexF64}, x::AbstractMatrix{ComplexF64},
                    ωₙ::AbstractVector{ComplexF64}, N::Integer)
    if N == 1
        y .= x
        return
    end
    N₂ = N >> 1
    yᵉᵛᵉⁿ = @view y[1:N₂, :]
    yᵒᵈᵈ = @view y[(N₂ + 1):end, :]
    xᵉᵛᵉⁿ = @view x[1:2:end, :]
    xᵒᵈᵈ = @view x[2:2:end, :]
    ωₙ₂ = @view ωₙ[1:2:end]
    apply_FFT!(yᵉᵛᵉⁿ, xᵉᵛᵉⁿ, ωₙ₂, N₂)
    apply_FFT!(yᵒᵈᵈ, xᵒᵈᵈ, ωₙ₂, N₂)
    yᵒᵈᵈ .*= ωₙ
    xₜₑₘₚ = @view x[1:N₂, :]
    @. xₜₑₘₚ = yᵉᵛᵉⁿ - yᵒᵈᵈ
    yᵉᵛᵉⁿ .+= yᵒᵈᵈ
    yᵒᵈᵈ .= xₜₑₘₚ
    nothing
end

N = 512
M = 1
x = rand(ComplexF64, N, M)
mine = apply_FFT(x)
fftw = fft(x)
diff = mine - fftw
norm(diff)

Ns = 1 .<< (1:10)
my_FFT_time = similar(Ns, Float64)
M = 1
for (i, N) in enumerate(Ns)
    println(i, " ", N)
    local x = rand(ComplexF64, N, M)
    my_FFT_time[i] = @belapsed apply_FFT($x)
end
plot!(plt_m1, Ns, my_FFT_time, label = "my FFT", markershape = :diamond)

M = 100
Ns = 1 .<< (1:10)
my_FFT_time = similar(Ns, Float64)
for (i, N) in enumerate(Ns)
    println(i, " ", N)
    local x = rand(ComplexF64, N, M)
    my_FFT_time[i] = @belapsed apply_FFT($x)
end
plot!(plt_m100, Ns, my_FFT_time, label = "my FFT", markershape = :diamond)

function apply_FFT_with_cached_DFT(x::AbstractMatrix)
    if N == 1
        return convert.(ComplexF64, x)
    end
    if N < 129 # 2⁷
        i = trailing_zeros(N)
        return DFT_cache[i] * x
    end
    x = convert.(ComplexF64, x)
    N = size(x, 1)
    y = similar(x)
    ωₙ = cispi.(range(0; step = -2 / N, length = N >> 1))
    apply_FFT_with_cached_DFT!(y, x, ωₙ, N)
    return y
end
function apply_FFT_with_cached_DFT!(y::AbstractMatrix{ComplexF64},
                                    x::AbstractMatrix{ComplexF64},
                                    ωₙ::AbstractVector{ComplexF64},
                                    N::Integer)
    if N < 129 # 2⁷
        i = trailing_zeros(N)
        mul!(y, DFT_cache[i], x)
        return
    end
    N₂ = N >> 1
    yᵉᵛᵉⁿ = @view y[1:N₂, :]
    yᵒᵈᵈ = @view y[(N₂ + 1):end, :]
    xᵉᵛᵉⁿ = @view x[1:2:end, :]
    xᵒᵈᵈ = @view x[2:2:end, :]
    ωₙ₂ = @view ωₙ[1:2:end]
    apply_FFT!(yᵉᵛᵉⁿ, xᵉᵛᵉⁿ, ωₙ₂, N₂)
    apply_FFT!(yᵒᵈᵈ, xᵒᵈᵈ, ωₙ₂, N₂)
    yᵒᵈᵈ .*= ωₙ
    xₜₑₘₚ = @view x[1:N₂, :]
    @. xₜₑₘₚ = yᵉᵛᵉⁿ - yᵒᵈᵈ
    yᵉᵛᵉⁿ .+= yᵒᵈᵈ
    yᵒᵈᵈ .= xₜₑₘₚ
    nothing
end

Ns = 1 .<< (1:10)
my_FFT_cached_DFT_time = similar(Ns, Float64)
M = 1
for (i, N) in enumerate(Ns)
    println(i, " ", N)
    local x = rand(ComplexF64, N, M)
    my_FFT_cached_DFT_time[i] = @belapsed apply_FFT_with_cached_DFT($x)
end
plot!(plt_m1, Ns, my_FFT_cached_DFT_time, label = "my FFT with cached DFT",
      markershape = :hexagon)

M = 100
Ns = 1 .<< (1:10)
my_FFT_cached_DFT_time = similar(Ns, Float64)
for (i, N) in enumerate(Ns)
    println(i, " ", N)
    local x = rand(ComplexF64, N, M)
    my_FFT_cached_DFT_time[i] = @belapsed apply_FFT_with_cached_DFT($x)
end
plot!(plt_m100, Ns, my_FFT_cached_DFT_time, label = "my FFT with cached DFT",
      markershape = :hexagon)

savefig(plt_m1, "time_m1.png")
savefig(plt_m100, "time_m100.png")
