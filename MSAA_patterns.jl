# DirectX MSAA subpixel offsets from Ray Tracing Gems II Chapter 3 Listing 3-1.
# We transpose and collect the matrices because Julia has major column memory alignment, so transpose to get the a 2xN matrix and collect to align memory
const MSAA_S1 = collect([0.5 0.5]')

const MSAA_S2 = collect([0.25 0.25; 0.75 0.75]')

const MSAA_S4 = collect([0.375 0.125; 0.875 0.375;
                         0.625 0.875; 0.125 0.625]')

const MSAA_S8 = collect([0.5625 0.6875; 0.4375 0.3125;
                         0.8125 0.4375; 0.3125 0.8125;
                         0.1875 0.1875; 0.0625 0.5625;
                         0.6875 0.0625; 0.9375 0.9375]')

const MSAA_S16 = collect([0.5625 0.4375; 0.4375 0.6875;
                          0.3125 0.3750; 0.7500 0.5625;
                          0.1875 0.6250; 0.6250 0.1875;
                          0.1875 0.3125; 0.6875 0.8125;
                          0.3750 0.1250; 0.5000 0.9375;
                          0.2500 0.8750; 0.1250 0.2500;
                          0.0000 0.5000; 0.9375 0.7500;
                          0.8750 0.0625; 0.0625 0.0000]')

# Code to generate Halton Sequence pixel offsets from Ray Tracing Gems II Chapter 3 Listing 3-2.
function generate_Halton_sequence!(N::Int, b::Int, sequence::AbstractVector{T}) where T<:AbstractFloat
    @assert b > 1
    n, d = 0, 1
    for i = 1:N
        x = d-n
        if x == 1
            n = 1
            d *= b
        else
            y = d ÷ b
            while x <= y 
                y ÷= b
            end
            n = (b + 1) * y - x
        end
        sequence[i] = T(n) / d
    end
    return sequence
end

function generate_Halton_sequence(N::Int, b::Int)
    sequence = Vector{Float32}(undef, N)
    generate_Halton_sequence!(N, b, sequence)
end

function generate_subpixel_offsets(N::Int; offset::Array{T, 2}=zeros(Float64, N, 2)) where T<:AbstractFloat
    generate_Halton_sequence!(N, 2, (@view offset[:, 1]))
    generate_Halton_sequence!(N, 3, (@view offset[:, 2]))
    offset .-= T(0.5)
end

function Halton_sequence32(::Type{T}, N::Int) where T <: AbstractFloat
    return collect(generate_subpixel_offsets(N; offset = zeros(T, N, 2))')
end

function Halton_sequence32(N::Int)
    return collect(generate_subpixel_offsets(N)')
end