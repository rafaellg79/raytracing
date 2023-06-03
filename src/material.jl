include("texture.jl")

@enum MaterialType begin
    Lambertian = 1
    Metal = 2
    Dielectric = 3
    Emissive = 4
    Isotropic = 5
end

struct Material{F, T}
    type::MaterialType
    albedo::T
    fuzz::F
    eta::F
end

Material{F, T}() where {F, T} = Material{F, T}(Lambertian, T(), zero(F), zero(F))
Material(type::MaterialType, albedo::Vec3{F}, fuzz::F=zero(F)) where F<:AbstractFloat = Material{F, SolidColor{F}}(type, SolidColor(albedo), fuzz, zero(F))
Material(type::MaterialType, albedo::T, fuzz::F) where {F<:AbstractFloat, T<:Texture} = Material{F, T}(type, albedo, fuzz, zero(F))
Material(type::MaterialType, eta::F) where F<:AbstractFloat = Material{F, SolidColor{F}}(type, SolidColor{F}(zero(Vec3{F})), zero(F), eta)

function reflect(v::Vec3{F}, n::Vec3{F}) where F<:AbstractFloat
    return v - 2 * dot(v, n) * n
end

function refract(direction::Vec3{F}, normal::Vec3{F}, etai_over_etat::F) where F<:AbstractFloat
    cos_theta = dot(-direction, normal)
    perp = etai_over_etat * (direction + cos_theta * normal)
    parallel = - sqrt(abs(one(F) - dot(perp, perp))) * normal
    return perp + parallel
end

function schlick(cos_theta::F, eta::F) where F<:AbstractFloat
    k = (1 - eta) / (1 + eta)
    r0 = k^2
    return r0 + (1 - r0) * ((1 - cos_theta)^5)
end