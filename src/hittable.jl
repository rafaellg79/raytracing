include("bounding_box.jl")
include("material.jl")
include("ray.jl")
include("hit_record.jl")

abstract type Hittable{T} end

textype(::Type{<:Hittable}) = Material
textype(::Type{<:Hittable{T}}) where T<:Hittable = textype(T)
textype(::Type{<:AbstractVector{T}}) where T<:Hittable = textype(T)
textype(::Type{<:AbstractVector{T}}) where T<:Material = T
textype(objects::Vector{T}) where T<:Hittable = reduce(promote_type, (tex for tex in textype.(typeof.(objects))))
textype(::Type{<:Hittable{T}}) where T<:Material = T

bounding_box(objects::AbstractVector{<:Hittable}, t::F) where F <: AbstractFloat = surrounding_box(objects)

function surrounding_box(objects::AbstractVector{<:Hittable})
    return reduce((a, b) -> AABB(min(a.min, b.min), max(a.max, b.max)), bounding_box(obj) for obj in objects)
end

function hit(objects::AbstractVector{T}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Hittable}
    closest_hit = HitRecord(F, textype(T))
    for i in 1:length(objects)
        h = hit(objects[i], r, t_min, t_max)
        if h.t < closest_hit.t
            closest_hit = h
            t_max = closest_hit.t
        end
    end
    return closest_hit
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