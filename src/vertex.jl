include("vec3.jl")

struct Vertex{F}
    position::Vec3{F}
    normal::Vec3{F}
    u::F
    v::F
end

Vertex(position::Vec3{F}) where F<:AbstractFloat = Vertex{F}(position, zero(Vec3{F}), F(-1), F(-1))
Vertex(position::Vec3{F}, normal::Vec3{F}) where F<:AbstractFloat = Vertex{F}(position, normal, F(-1), F(-1))
Vertex(position::Vec3{F}, u::F, v::F) where F<:AbstractFloat = Vertex{F}(position, zero(Vec3{F}), u, v)
Vertex(position::Vec3{F}, normal::Vec3{F}, u::F, v::F) where F<:AbstractFloat = Vertex{F}(position, normal, u, v)

import Base: +, -, *, eltype

eltype(v::Vertex{F}) where F<:AbstractFloat = F
eltype(::Type{Vertex{F}}) where F<:AbstractFloat = F

+(a::V, b::V) where V<:Vertex = Vertex(a.position + b.position, a.normal + b.normal, a.u + b.u, a.v + b.v)
-(a::V, b::V) where V<:Vertex = Vertex(a.position - b.position, a.normal - b.normal, a.u - b.u, a.v - b.v)
*(a::V, b::V) where V<:Vertex = Vertex(a.position * b.position, a.normal * b.normal, a.u * b.u, a.v * b.v)

*(a::V, b) where V<:Vertex = Vertex(a.position * b, a.normal * b, a.u * b, a.v * b)
*(a, b::V) where V<:Vertex = Vertex(a * b.position, a * b.normal, a * b.u, a * b.v)