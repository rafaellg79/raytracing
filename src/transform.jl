include("hittable.jl")

struct Transform{F, T} <: Hittable{T}
    object::T
    xform::NTuple{12, F}
    inv_xform::NTuple{12, F}
    bbox::AABB{F}
end

import Adapt, Base.eltype

function Adapt.adapt_structure(to, transform::Transform)
    object = Adapt.adapt_structure(to, transform.object)
    Transform(object, transform.xform, transform.inv_xform, transform.bbox)
end

eltype(::Type{Transform{F, T}}) where {F, T} = T

function Transform(transform::Transform{F}, xform::Matrix{F}) where F<:AbstractFloat
    xform *= transform.xform
    inv_xform = inv(xform)
    
    bbox = bounding_box(transform.object)
    
    min_x, min_y, min_z = Inf, Inf, Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    
    for k = 0:1, j = 0:1, i = 0:1
        v = Vec3{F}(
                i * bbox.max.x + (1-i) * bbox.min.x,
                j * bbox.max.y + (1-j) * bbox.min.y,
                k * bbox.max.z + (1-k) * bbox.min.z
            )
        
        t_v = apply_point(xform, v)
        
        min_x, min_y, min_z = min(t_v.x, min_x), min(t_v.y, min_y), min(t_v.z, min_z)
        max_x, max_y, max_z = max(t_v.x, max_x), max(t_v.y, max_y), max(t_v.z, max_z)
    end
    
    return Transform(transform.object, tuple(xform[1:3, 1:4]...), tuple(inv_xform[1:3, 1:4]...), AABB{F}(Vec3{F}(min_x, min_y, min_z), Vec3{F}(max_x, max_y, max_z)))
end

function Transform(object::T, xform::Matrix{F}) where {F<:AbstractFloat, T<:Hittable}
    inv_xform = inv(xform)
    
    bbox = bounding_box(object)
    
    min_x, min_y, min_z = Inf, Inf, Inf
    max_x, max_y, max_z = -Inf, -Inf, -Inf
    
    for k = 0:1, j = 0:1, i = 0:1
        v = Vec3{F}(
                i * bbox.max.x + (1-i) * bbox.min.x,
                j * bbox.max.y + (1-j) * bbox.min.y,
                k * bbox.max.z + (1-k) * bbox.min.z
            )
        
        t_v = apply_point(xform, v)
        
        min_x, min_y, min_z = min(t_v.x, min_x), min(t_v.y, min_y), min(t_v.z, min_z)
        max_x, max_y, max_z = max(t_v.x, max_x), max(t_v.y, max_y), max(t_v.z, max_z)
    end
    
    return Transform(object, tuple(xform[1:3, 1:4]...), tuple(inv_xform[1:3, 1:4]...), AABB{F}(Vec3{F}(min_x, min_y, min_z), Vec3{F}(max_x, max_y, max_z)))
end

function apply_point(T::Matrix{F}, v::Vec3{F}) where F<:AbstractFloat
    return Vec3{F}(
            T[1, 1] * v.x + T[1, 2] * v.y + T[1, 3] * v.z + T[1, 4],
            T[2, 1] * v.x + T[2, 2] * v.y + T[2, 3] * v.z + T[2, 4],
            T[3, 1] * v.x + T[3, 2] * v.y + T[3, 3] * v.z + T[3, 4]
           )
end

function apply_point(T::Tuple, v::Vec3{F}) where F<:AbstractFloat
    return Vec3{F}(
            T[1] * v.x + T[4] * v.y + T[7] * v.z + T[10],
            T[2] * v.x + T[5] * v.y + T[8] * v.z + T[11],
            T[3] * v.x + T[6] * v.y + T[9] * v.z + T[12]
           )
end

function apply_vec(T::Matrix{F}, v::Vec3{F}) where F<:AbstractFloat
    return Vec3{F}(
            T[1, 1] * v.x + T[1, 2] * v.y + T[1, 3] * v.z,
            T[2, 1] * v.x + T[2, 2] * v.y + T[2, 3] * v.z,
            T[3, 1] * v.x + T[3, 2] * v.y + T[3, 3] * v.z
           )
end

function apply_vec(T::Tuple, v::Vec3{F}) where F<:AbstractFloat
    return Vec3{F}(
            T[1] * v.x + T[4] * v.y + T[7] * v.z,
            T[2] * v.x + T[5] * v.y + T[8] * v.z,
            T[3] * v.x + T[6] * v.y + T[9] * v.z
           )
end

function bounding_box(transform::Transform)
    return transform.bbox
end

function hit(transform::Transform{F}, r::Ray{F}, t_min::F, t_max::F) where F<:AbstractFloat
    hit_result = hit(transform.object, Ray(apply_point(transform.inv_xform, r.origin), apply_vec(transform.inv_xform, r.direction), r.t), t_min, t_max)
    
    if hit_result.t == F(Inf)
        return hit_result
    end
    
    t = hit_result.t
    p = at(r, t)
    normal = apply_vec(transform.xform, hit_result.normal)
    
    return HitRecord(t, p, normal, hit_result.material, hit_result.front_face, hit_result.u, hit_result.v)
end