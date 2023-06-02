include("hittable.jl")

struct ConstantMedium{F, M, T} <: Hittable{T}
    boundary::T
    phase_function::M
    neg_inv_density::F
    ConstantMedium{F, M, T}(boundary::T, density::Real, phase_function::M) where {F<:AbstractFloat, M<:Material, T<:Hittable} = new(boundary, phase_function, -1 / F(density))
end

import Adapt, Base.eltype

function Adapt.adapt_structure(to, constant_medium::ConstantMedium)
    boundary = Adapt.adapt_structure(to, constant_medium.boundary)
    phase_function = Adapt.adapt_structure(to, constant_medium.phase_function)
    ConstantMedium(boundary, phase_function, constant_medium.neg_inv_density)
end

eltype(::Type{ConstantMedium{F, M, T}}) where {F, M, T} = T
textype(::Type{ConstantMedium{F, M, T}}) where {F, M, T} = M

function ConstantMedium(boundary::T, density::Real, color::Vec3{F}) where {F<:AbstractFloat, T<:Hittable}
    ConstantMedium{F, Material{F, SolidColor{F}}, T}(boundary, density, Material(Isotropic, color))
end

function bounding_box(c::ConstantMedium)
    bounding_box(c.boundary)
end

function hit(c::ConstantMedium{F, M, T}, r::Ray{F}, t_min::F, t_max::F) where {F, M, T}
    t1 = hit(c.boundary, r, F(-Inf), F(Inf)).t
    if t1 == F(Inf)
        return HitRecord(F, M)
    end
    
    t2 = hit(c.boundary, r, t1+F(0.0001), F(Inf)).t
    if t2 == F(Inf)
        return HitRecord(F, M)
    end
    
    if t1 < t_min
        t1 = t_min
    end
    if t2 > t_max
        t2 = t_max
    end
    if t1 >= t2
        return HitRecord(F, M)
    end
    
    if t1 < 0
        t1 = 0
    end
    
    ray_length = length(r.direction)
    distance_inside_boundary = (t2 - t1) * ray_length
    hit_distance = c.neg_inv_density * log(rand(F))
    
    if hit_distance > distance_inside_boundary
        return HitRecord(F, M)
    end
    
    t = t1 + hit_distance / ray_length
    p = at(r, t)
    return HitRecord(t, p, Vec3{F}(1, 0, 0), c.phase_function, false, zero(F), zero(F))
end