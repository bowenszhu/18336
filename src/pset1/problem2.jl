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

N = 512 << 1
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
              axis = :log10, minorgrid = true, markershape = :circle, title = "M = $M")
plot!(plt_m1, Ns, FFT_time, label = "FFT", markershape = :circle)

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
plot!(plt_m1, Ns, cached_DFT_time, label = "DFT cached", markershape = :circle)
savefig(plt_m1, "time_m1.png")

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
                axis = :log10, minorgrid = true, markershape = :circle, title = "M = $M")
plot!(plt_m100, Ns, FFT_time, label = "FFT", markershape = :circle)
plot!(plt_m100, Ns, cached_DFT_time, label = "DFT cached", markershape = :circle)
savefig(plt_m100, "time_m100.png")
