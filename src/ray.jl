include("vec3.jl")

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

function at(ray::Ray{F}, t::F) where F
    return ray.origin + ray.direction * t
end