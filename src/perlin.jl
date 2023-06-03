using Random

include("vec3.jl")

struct Perlin3{T}
    random::Vector{T}
    perm_x::Vector{Int}
    perm_y::Vector{Int}
    perm_z::Vector{Int}
end

const Perlin3f = Perlin3{Float32}
const Perlin3v = Perlin3{Vec3{Float32}}

Perlin3{T}() where T = Perlin3{T}(T[], Int[], Int[], Int[])

function Perlin3{Vec3{F}}(point_count::Int) where F<:AbstractFloat
    random = [normalize(rand(Vec3{F}, -one(F), one(F))) for i = 1:point_count]
    perm_x = collect(0:point_count-1)
    perm_y = shuffle(perm_x)
    perm_z = shuffle(perm_x)
    shuffle!(perm_x)
    return Perlin3{Vec3{F}}(random, perm_x, perm_y, perm_z)
end

function Perlin3{T}(point_count::Int) where T
    random = rand(T, point_count)
    perm_x = collect(0:point_count-1)
    perm_y = shuffle(perm_x)
    perm_z = shuffle(perm_x)
    shuffle!(perm_x)
    return Perlin3{T}(random, perm_x, perm_y, perm_z)
end

function turbulence(tex::Perlin3{T}, p::Vec3{F}; depth::Int=7) where {F<:AbstractFloat, T}
    accum = zero(F)
    weight = one(F)
    
    for i = 1:depth
        accum += weight * noise(tex, p)
        weight *= F(0.5)
        p *= F(2)
    end
    
    return abs(accum)
end

function noise(perlin::Perlin3{T}, p::Vec3) where T
    u = p.x - floor(p.x)
    v = p.y - floor(p.y)
    w = p.z - floor(p.z)
    
    i = floor(Int, p.x)
    j = floor(Int, p.y)
    k = floor(Int, p.z)
    
    c = Array{T, 3}(undef, 2, 2, 2)
    
    for dk=1:2, dj=1:2, di=1:2
        c[di, dj, dk] = perlin.random[
            xor(perlin.perm_x[(i+di) & 255 + 1], 
                perlin.perm_y[(j+dj) & 255 + 1], 
                perlin.perm_z[(k+dk) & 255 + 1]
            ) + 1]
    end
    
    return trilinear_interp(c, u, v, w)
end

function trilinear_interp(c::Array{Vec3{F}}, u::Real, v::Real, w::Real) where F<:AbstractFloat
    uu = F(u * u * (3 - 2u))
    vv = F(v * v * (3 - 2v))
    ww = F(w * w * (3 - 2w))
    
    accum = zero(F)
    for k=1:2, j=1:2, i=1:2
        weight_v = Vec3{F}(u-i, v-j, w-k)
        accum += (i*uu + (2-i)*(1-uu)) *
                 (j*vv + (2-j)*(1-vv)) *
                 (k*ww + (2-k)*(1-ww)) *
                 dot(c[i, j, k], weight_v)
    end

    return accum
end

function trilinear_interp(c::Array{T}, u::T, v::T, w::T) where T
    accum = zero(T)
    for k=1:2, j=1:2, i=1:2
        accum += (i*u + (2-i)*(1-u)) *
                 (j*v + (2-j)*(1-v)) *
                 (k*w + (2-k)*(1-w)) *
                 c[i, j, k]
    end

    return accum
end