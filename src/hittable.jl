include("bounding_box.jl")
include("material.jl")

struct Ray{F}
    origin::Vec3{F}
    direction::Vec3{F}
    t::F
    Ray{F}(origin::Vec3{F}, direction::Vec3{F}) where F = new(origin, direction, zero(F))
    Ray{F}(origin::Vec3{F}, direction::Vec3{F}, t::F) where F = new(origin, direction, t)
end

Ray(origin::Vec3{F}, direction::Vec3{F}) where F = Ray{F}(origin, direction)
Ray(origin::Vec3{F}, direction::Vec3{F}, t::F) where F = Ray{F}(origin, direction, t)
Ray(origin::Vec3{F}, direction::Vec3{F}, t::Real) where F = Ray{F}(origin, direction, F(t))

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

abstract type Hittable{T} end

textype(::Type{<:Hittable}) = Material
textype(::Type{<:Hittable{T}}) where T<:Hittable = textype(T)
textype(::Type{<:AbstractVector{T}}) where T<:Hittable = textype(T)
textype(objects::Vector{T}) where T<:Hittable = reduce(promote_type, (tex for tex in textype.(typeof.(objects))))
textype(::Type{<:Hittable{T}}) where T<:Material = T

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

bounding_box(objects::Vector{<:Hittable}, t::F) where F <: AbstractFloat = surrounding_box(list)

function surrounding_box(objects::AbstractVector{<:Hittable})
    return reduce((a, b) -> AABB(min(a.min, b.min), max(a.max, b.max)), bounding_box(obj) for obj in objects)
end

function get_sphere_uv(p::Vec3{F}) where F<:AbstractFloat
    theta = acos(clamp(-p.y, -one(F), one(F)))
    phi = (atan(-p.z, p.x) + F(pi))
    return phi / F(2pi), theta / F(pi)
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

function hit(bbox::AABB, r::Ray{F}, t_min::F, t_max::F) where F <: AbstractFloat
    tmin = (bbox.min.x - r.origin.x) / r.direction.x
    tmax = (bbox.max.x - r.origin.x) / r.direction.x
    if tmin > tmax
        tmin, tmax = tmax, tmin
    end
    if tmax < t_max
        t_max = tmax
    end
    if tmin > t_min
        t_min = tmin
    end
    if t_max <= t_min
        return false
    end

    tmin = (bbox.min.y - r.origin.y) / r.direction.y
    tmax = (bbox.max.y - r.origin.y) / r.direction.y
    if tmin > tmax
        tmin, tmax = tmax, tmin
    end
    if tmax < t_max
        t_max = tmax
    end
    if tmin > t_min
        t_min = tmin
    end
    if t_max <= t_min
        return false
    end

    tmin = (bbox.min.z - r.origin.z) / r.direction.z
    tmax = (bbox.max.z - r.origin.z) / r.direction.z
    if tmin > tmax
        tmin, tmax = tmax, tmin
    end
    if tmax < t_max
        t_max = tmax
    end
    if tmin > t_min
        t_min = tmin
    end
    if t_max <= t_min
        return false
    end
    
    return true
end

function hit(objects::Array{T}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Hittable}
    closest_t = t_max
    closest_ind = 0
    for i in 1:length(objects)
        t = hit(objects[i], r, t_min, closest_t)
        if t < closest_t
            closest_t = t
            closest_ind = i
        end
    end
    if closest_ind == 0
        return HitRecord(F, textype(T))
    end
    return HitRecord(objects[closest_ind], r, closest_t)
end

hit(obj::T, r::Ray{F}) where {F <: AbstractFloat, T <: Hittable} = hit(obj, r, F(0.001), F(Inf))
hit(objects::Array{T}, r::Ray{F}) where {F <: AbstractFloat, T <: Hittable} = hit(objects, r, F(0.001), F(Inf))

function scatter(ray::Ray{F}, hit::HitRecord{F, T}) where {F<:AbstractFloat, T<:Material{F}}
    material = hit.material
    if material.type == Lambertian
        scatter_dirrection = hit.normal + random_unit_vector(Vec3{F})
        if almost_zero(scatter_dirrection)
            scatter_dirrection = hit.normal
        end
        scattered = Ray(hit.p, scatter_dirrection, ray.t)
        attenuation = value(material.albedo, hit.u, hit.v, hit.p)
        return scattered, attenuation
    elseif material.type == Metal
        reflected = reflect(normalize(ray.direction), hit.normal)
        scattered = Ray(hit.p, reflected + material.fuzz * random_in_unit_sphere(Vec3{F}), ray.t)
        attenuation = (dot(scattered.direction, hit.normal) > zero(F)) * value(material.albedo, hit.u, hit.v, hit.p)
        return scattered, attenuation
    elseif material.type == Dielectric
        refraction_ratio = hit.front_face ? (one(F)/material.eta) : material.eta
        unit_dir = normalize(ray.direction)
        cos_theta = min(dot(-unit_dir, hit.normal), one(F))
        sin_theta = sqrt(one(F) - cos_theta*cos_theta)
        
        scatter_direction = zero(Vec3{F})
        
        if (refraction_ratio * sin_theta > one(F)) || (schlick(cos_theta, material.eta) > rand(F))
            scatter_direction = reflect(unit_dir, hit.normal)
        else
            scatter_direction = refract(unit_dir, hit.normal, refraction_ratio)
        end
        scattered = Ray(hit.p, scatter_direction, ray.t)
        
        attenuation = Vec3{F}(one(F))
        return scattered, attenuation
    elseif material.type == Isotropic
        scattered = Ray(hit.p, random_in_unit_sphere(Vec3{F}), ray.t)
        attenuation = value(material.albedo, hit.u, hit.v, hit.p)
        return scattered, attenuation
    else
        return Ray(zero(Vec3{F}), zero(Vec3{F})), zero(Vec3{F})
    end
end

function emit(hit::HitRecord{F, T}) where {F<:AbstractFloat, T<:Material}
    if hit.material.type == Emissive
        return value(hit.material.albedo, hit.u, hit.v, hit.p)
    else
        return zero(Vec3{F})
    end
end