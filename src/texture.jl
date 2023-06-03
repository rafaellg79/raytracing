using Images

include("vec3.jl")
include("perlin.jl")

abstract type Texture end

struct SolidColor{F} <: Texture
    color::Vec3{F}
    SolidColor{F}(color::Vec3{F}) where F<:AbstractFloat = new(color)
    SolidColor{F}() where F<:AbstractFloat = new(zero(Vec3{F}))
end

SolidColor(::Type{F}) where F<:AbstractFloat = SolidColor(zero(Vec3{F}))
SolidColor(color::Vec3{F}) where F<:AbstractFloat = SolidColor{F}(color)
SolidColor(r::F, g::F, b::F) where F<:AbstractFloat = SolidColor{F}(Vec3{F}(r, g, b))

struct CheckerTexture{Odd, Even} <: Texture
    odd::Odd
    even::Even
end

struct ImageTexture <: Texture
    image::Matrix{RGB}
end

ImageTexture() = ImageTexture(Matrix{RGB}(undef, 0, 0))
ImageTexture(filename::String) = ImageTexture(load(filename))

mutable struct NoiseTexture{T} <: Texture
    noise::Perlin3{T}
    scale::Float64
end

NoiseTexture{T}() where T = NoiseTexture{T}(Perlin3{T}(), 1.0)
NoiseTexture{T}(noise::Perlin3{T}) where T = NoiseTexture(noise, 1.0)
NoiseTexture{T}(scale::Real; point_count::Int=256) where T = NoiseTexture(Perlin3{T}(point_count), Float64(scale))

#TODO: Implement CUDA versions of CheckerTexture, ImageTexture and NoiseTexture.
#      The Texture Memory is still experimental in the CUDA.jl API so expect changes in the API.

function value(tex::SolidColor, u::F, v::F, p::Vec3{F}) where F<:AbstractFloat
    return tex.color
end

function value(tex::CheckerTexture, u::Real, v::Real, p::Vec3{F}) where F<:AbstractFloat
    sines = sin(10 * p.x) * sin(10*p.y) * sin(10 * p.z)
    if sines < 0
        return value(tex.odd, u, v, p)
    else
        return value(tex.even, u, v, p)
    end
end

function value(tex::ImageTexture, u::Real, v::Real, p::Vec3{F}) where F<:AbstractFloat
    if tex.image == []
        return Vec3{F}(0, 1, 1)
    end
    
    u = clamp01(u)
    v = 1 - clamp01(v)
    
    i = trunc(Int, v * size(tex.image, 1))
    j = trunc(Int, u * size(tex.image, 2))
    
    if i == 0
        i = 1
    end
    if j == 0
        j = 1
    end
    
    color = tex.image[i, j]
    
    return Vec3{F}(color.r, color.g, color.b)
end

# Perlin noise (floats)
function value(tex::NoiseTexture{F}, u::Real, v::Real, p::Vec3{F}) where F<:AbstractFloat
    return Vec3{F}(1) * noise(tex.noise, tex.scale * p)
end

#=
# Perlin noise (random unit vectors)
function value(tex::NoiseTexture{Vec3{F}}, u::Real, v::Real, p::Vec3{F}) where F<:AbstractFloat
    return Vec3{F}(1) * 0.5 * (1 + noise(tex.noise, tex.scale * p))
end


# Turbulence
function value(tex::NoiseTexture{Vec3{F}}, u::Real, v::Real, p::Vec3{F}) where F<:AbstractFloat
    return Vec3{F}(1) * turbulence(tex.noise, tex.scale * p)
end
=#


# Marble like
function value(tex::NoiseTexture{Vec3{F}}, u::Real, v::Real, p::Vec3{F}) where F
    return Vec3{F}(1) * F(0.5) * (1 + sin(tex.scale*p.z + 10turbulence(tex.noise, p)))
end