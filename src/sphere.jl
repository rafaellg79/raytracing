include("hittable.jl")

struct Sphere{F, T} <: Hittable{T}
    center::Vec3{F}
    radius::F
    material::T
    Sphere{F, T}(center::Vec3{F}, radius::F, material::T) where {F <:AbstractFloat, T<:Material{F}} = new(center, radius, material)
end

Sphere(center::Vec3{F}, radius::Real, material::T) where {F<:AbstractFloat, T<:Material} = Sphere{F, T}(center, F(radius), material)

struct MovingSphere{F, T} <: Hittable{T}
    center0::Vec3{F}
    t0::F
    center1::Vec3{F}
    t1::F
    radius::F
    material::T
end

MovingSphere(center0::Vec3{F}, center1::Vec3{F}, t0::Real, t1::Real, radius::Real, material::T) where {F<:AbstractFloat, T<:Material} = MovingSphere{F, T}(center0, F(t0), center1, F(t1), F(radius), material)
MovingSphere(center::Vec3{F}, radius::Real, material::T) where {F<:Real, T<:Material} = MovingSphere{F, T}(center, zero(F), center, one(F), F(radius), material)

function center_at(s::MovingSphere{F}, t::F) where F
    return s.center0 + ((t - s.t0) / (s.t1 - s.t0)) * (s.center1 - s.center0)
end

function at(ray::Ray{F}, t::F) where F
    return ray.origin + ray.direction * t
end

function bounding_box(s::Sphere{F}) where F<:AbstractFloat
    return AABB{F}(s.center-abs(s.radius), s.center+abs(s.radius))
end

function bounding_box(s::MovingSphere{F}) where F<:AbstractFloat
    return AABB{F}(min(s.center0-abs(s.radius), s.center1-abs(s.radius)), max(s.center0+abs(s.radius), s.center1+abs(s.radius)))
end

function bounding_box(s::MovingSphere{F}, t::F) where F
    center = center_at(s, t)
    return AABB{F}(center-abs(s.radius), center+abs(s.radius))
end

function hit(s::Sphere{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    oc = r.origin - s.center
    a = dot(r.direction, r.direction)
    half_b = dot(oc, r.direction)
    c = dot(oc, oc) - s.radius * s.radius
    
    discriminant = half_b*half_b - a*c
    if discriminant < zero(F)
        return HitRecord(F, T)
    end
    sqrtd = sqrt(discriminant)
    
    root = (-half_b - sqrtd)
    if root < t_min * a || t_max * a < root
        root = (-half_b + sqrtd)
        if root < t_min * a || t_max * a < root
            return HitRecord(F, T)
        end
    end
    
    root /= a
    
    p = at(r, root)
    normal = (p - s.center) / s.radius
    front_face = dot(r.direction, normal) < zero(F)
    u, v = get_sphere_uv(normal)
    return HitRecord(root, p, (-1+(front_face<<1)) * normal, s.material, front_face, u, v)
end

function hit(s::MovingSphere{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    center = center_at(s, r.t)
    oc = r.origin - center
    a = dot(r.direction, r.direction)
    half_b = dot(oc, r.direction)
    c = dot(oc, oc) - s.radius * s.radius
    
    discriminant = half_b*half_b - a*c
    if discriminant < zero(F)
        return HitRecord(F, T)
    end
    sqrtd = sqrt(discriminant)
    
    root = (-half_b - sqrtd)
    if root < t_min * a || t_max * a < root
        root = (-half_b + sqrtd)
        if root < t_min * a || t_max * a < root
            return HitRecord(F, T)
        end
    end
    
    root /= a
    
    p = at(r, root)
    normal = (p - center) / s.radius
    front_face = dot(r.direction, normal) < zero(F)
    u, v = get_sphere_uv(normal)
    return HitRecord(root, p, (-1+(front_face<<1)) * normal, s.material, front_face, u, v)
end

function get_sphere_uv(p::Vec3{F}) where F<:AbstractFloat
    theta = acos(clamp(-p.y, -one(F), one(F)))
    phi = (atan(-p.z, p.x) + F(pi))
    return phi / F(2pi), theta / F(pi)
end