include("vec3.jl")
include("material.jl")

struct HitRecord{F, T}
    t::F
    p::Vec3{F}
    normal::Vec3{F}
    material::T
    front_face::Bool
    u::F
    v::F
end

HitRecord(::Type{F}, ::Type{T}) where {F<:AbstractFloat, T<:Material} = HitRecord{F, T}(F(Inf), Vec3{F}(F(Inf)), Vec3{F}(zero(F)), T(), false, zero(F), zero(F))
HitRecord(t::F, p::Vec3{F}, normal::Vec3{F}, material::T, front_face::Bool, u::F, v::F) where {F<:AbstractFloat, T<:Material} = HitRecord{F, T}(t, p, normal, material, front_face, u, v)