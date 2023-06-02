include("hittable.jl")

struct xy_rect{F, T} <: Hittable{T}
    x0::F
    x1::F
    y0::F
    y1::F
    k::F
    material::T
    xy_rect{F, T}(x0::Real, x1::Real, y0::Real, y1::Real, k::Real, material::T) where T<:Material{F} where F<:AbstractFloat = new(F(x0), F(x1), F(y0), F(y1), F(k), material)
end

struct xz_rect{F, T} <: Hittable{T}
    x0::F
    x1::F
    z0::F
    z1::F
    k::F
    material::T
    xz_rect{F, T}(x0::Real, x1::Real, z0::Real, z1::Real, k::Real, material::T) where T<:Material{F} where F<:AbstractFloat = new(F(x0), F(x1), F(z0), F(z1), F(k), material)
end

struct yz_rect{F, T} <: Hittable{T}
    y0::F
    y1::F
    z0::F
    z1::F
    k::F
    material::T
    yz_rect{F, T}(y0::Real, y1::Real, z0::Real, z1::Real, k::Real, material::T) where T<:Material{F} where F<:AbstractFloat = new(F(y0), F(y1), F(z0), F(z1), F(k), material)
end

xy_rect(x0::Real, x1::Real, y0::Real, y1::Real, k::Real, material::T) where T<:Material{F} where F = xy_rect{F, T}(F(x0), F(x1), F(y0), F(y1), F(k), material)
xz_rect(x0::Real, x1::Real, z0::Real, z1::Real, k::Real, material::T) where T<:Material{F} where F = xz_rect{F, T}(F(x0), F(x1), F(z0), F(z1), F(k), material)
yz_rect(y0::Real, y1::Real, z0::Real, z1::Real, k::Real, material::T) where T<:Material{F} where F = yz_rect{F, T}(F(y0), F(y1), F(z0), F(z1), F(k), material)

function bounding_box(rec::xy_rect{F}) where F<:AbstractFloat
    return AABB{F}(Vec3{F}(rec.x0, rec.y0, rec.k - F(0.0001)), Vec3{F}(rec.x1, rec.y1, rec.k + F(0.0001)))
end

function bounding_box(rec::xz_rect{F}) where F<:AbstractFloat
    return AABB{F}(Vec3{F}(rec.x0, rec.k - F(0.0001), rec.z0), Vec3{F}(rec.x1, rec.k + F(0.0001), rec.z1))
end

function bounding_box(rec::yz_rect{F}) where F<:AbstractFloat
    return AABB{F}(Vec3{F}(rec.k - F(0.0001), rec.y0, rec.z0), Vec3{F}(rec.k + F(0.0001), rec.y1, rec.z1))
end

function hit(rec::xy_rect{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F, T}
    t = (rec.k-r.origin.z) / r.direction.z
    if t < t_min || t_max < t
        return HitRecord(F, T)
    end
    
    x = r.origin.x + t * r.direction.x
    y = r.origin.y + t * r.direction.y
    if x < rec.x0 || rec.x1 < x || y < rec.y0 || rec.y1 < y
        return HitRecord(F, T)
    end
    
    p = Vec3{F}(x, y, rec.k)
    return HitRecord(rec, r, t, p)
end

function hit(rec::xz_rect{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F, T}
    t = (rec.k-r.origin.y) / r.direction.y
    if t < t_min || t_max < t
        return HitRecord(F, T)
    end
    
    x = r.origin.x + t * r.direction.x
    z = r.origin.z + t * r.direction.z
    if x < rec.x0 || rec.x1 < x || z < rec.z0 || rec.z1 < z
        return HitRecord(F, T)
    end
    
    p = Vec3{F}(x, rec.k, z)
    return HitRecord(rec, r, t, p)
end

function hit(rec::yz_rect{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F, T}
    t = (rec.k-r.origin.x) / r.direction.x
    if t < t_min || t_max < t
        return HitRecord(F, T)
    end
    
    y = r.origin.y + t * r.direction.y
    z = r.origin.z + t * r.direction.z
    if y < rec.y0 || rec.y1 < y || z < rec.z0 || rec.z1 < z
        return HitRecord(F, T)
    end
    
    p = Vec3{F}(rec.k, y, z)
    return HitRecord(rec, r, t, p)
end

function HitRecord(rec::xy_rect{F, T}, r::Ray{F}, t::F, p::Vec3{F}) where {F, T}
    normal = Vec3{F}(0, 0, 1)
    front_face = dot(r.direction, normal) < 0
    u = (p.x - rec.x0)/(rec.x1 - rec.x0)
    v = (p.y - rec.y0)/(rec.y1 - rec.y0)
    return HitRecord(t, p, (-1 + 2 * front_face) * normal, rec.material, front_face, u, v)
end

function HitRecord(rec::xz_rect{F, T}, r::Ray{F}, t::F, p::Vec3{F}) where {F, T}
    normal = Vec3{F}(0, 1, 0)
    front_face = dot(r.direction, normal) < 0
    u = (p.x - rec.x0)/(rec.x1 - rec.x0)
    v = (p.z - rec.z0)/(rec.z1 - rec.z0)
    return HitRecord(t, p, (-1 + 2 * front_face) * normal, rec.material, front_face, u, v)
end

function HitRecord(rec::yz_rect{F, T}, r::Ray{F}, t::F, p::Vec3{F}) where {F, T}
    normal = Vec3{F}(1, 0, 0)
    front_face = dot(r.direction, normal) < 0
    u = (p.y - rec.y0)/(rec.y1 - rec.y0)
    v = (p.z - rec.z0)/(rec.z1 - rec.z0)
    return HitRecord(t, p, (-1 + 2 * front_face) * normal, rec.material, front_face, u, v)
end