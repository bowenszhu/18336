using FFTW: plan_r2r!, REDFT00
using Plots: plot, savefig
using LaTeXStrings: @L_str

N = 1000
θₙ = 0:(1 / (N - 1)):1
xₙ = cospi.(θₙ)

# Problem 1.1
dct1! = plan_r2r!(xₙ, REDFT00)
function chebyshev!(u)
    dct1! * u
end
function backward!(u)
    dct1! * u
    u ./= 2(N - 1)
end
u1 = rand(N)
u2 = copy(u1)
backward!(chebyshev!(u2))
u1 ≈ u2

# Problem 1.2
f1(x) = +(sincospi(2 * x)...)
f2(x) = +(sincospi(200 * x)...)
f3(x) = √(1 - x^2)
f4(x) = 1 / (1 + 100 * (x - 0.1)^2)
f5(x) = abs(x - 0.5)^3
f6(x) = f2(x) * f4(x)

for f in (f1, f2, f3, f4, f5, f6)
    u = f.(xₙ)
    chebyshev!(u) # û
    @. u = log10(abs(u)) # log10 magnitude
    plt = plot(0:(N - 1), u, xlabel = L"n", ylabel = L"\log_{10}|\hat u_n|", title = "$f",
               legend = false)
    display(plt)
    savefig(plt, "p12_$f.svg")
end
