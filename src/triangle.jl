include("hittable.jl")

struct Triangle{F, T} <: Hittable{T}
    vertices::Tuple{Vec3{F}, Vec3{F}, Vec3{F}}
    material::T
end

Triangle(vertices::Vector{Vec3{F}}, material::T) where {F <:AbstractFloat, T<:Material{F}} = Triangle(Tuple(vertices), material)
Triangle(v1::Vec3{F}, v2::Vec3{F}, v3::Vec3{F}, material::T) where {F <:AbstractFloat, T<:Material{F}} = Triangle((v1, v2, v3), material)

function bounding_box(t::Triangle{F}) where F<:AbstractFloat
    return AABB{F}(minimum(t.vertices)-F(0.001), maximum(t.vertices)+F(0.001))
end

# Based on Möller–Trumbore intersection algorithm:
# Any point in a triangle can be represented as a combination of its vertices
# T(w, u, v) = wV1 + uV2 + vV3
#
# in particular if w=1-u-v we have
# T(u, v) = V1 + u*e1 + v*e2 where e1 and e2 are the edges e1 = V2-V1 and e2 = V3-V1
#
# we want to find the point where the ray hits the triangle that is
# ray_origin + t*ray_direction = V1 + u*e1 + v*e2 (1)
#
# applying the dot product with the normal of the triangle we obtain
# dot(normal, ray_origin + t*ray_direction) = dot(normal, V1 + u*e1 + v*e2)
#
# the dot product is distributive over vector addition so we can expand the equation into
# dot(normal, ray_origin) + dot(normal, t*ray_direction) = dot(normal, V1) + dot(normal, u*e1 + v*e2)
#
# note that dot(normal, u*e1 + v*e2) = 0 because the normal is orthogonal to both edges then
# dot(normal, ray_origin) + dot(normal, t*ray_direction) = dot(normal, V1)
#
# finally we can isolate t to get
# t = (dot(normal, V1) - dot(normal, ray_origin))/dot(normal, ray_direction)
#
# using the distributive property of dot product to join the numerator we can simplify the equation to
# t = dot(normal, V1 - ray_origin)/dot(normal, ray_direction)
#
# Repeat the process multiplying (1) by cross(e1, ray_direction) and cross(e2, ray_direction)
# observing that they are orthogonal to u*e1 and v*e2, respectively, to obtain
# u = dot(cross(e2, ray_direction), ray_origin - V1) / dot(cross(e2, ray_direction), e1)
# v = dot(cross(e1, ray_direction), ray_origin - V1) / dot(cross(e1, ray_direction), e2)
#
# the triple product has the following property
# dot(cross(a, b), c) = dot(cross(b, c), a) = dot(cross(c, a), b)
#
# thus we can rearrange the equations into
# u = dot(cross(ray_direction, ray_origin - V1), e2)/dot(cross(e1, e2), ray_direction)
# v = dot(cross(ray_direction, ray_origin - V1), e1)/dot(cross(e2, e1), ray_direction)
#
# remember that normal = cross(e1, e2) and note that cross(a, b) = -cross(b, a) to obtain
# u = dot(cross(ray_direction, ray_origin - V1), e2)/dot( normal, ray_direction)
# v = dot(cross(ray_direction, ray_origin - V1), e1)/dot(-normal, ray_direction)
#
# Let s = cross(ray_direction, ray_origin - V1) and simplify the equations to
# u = dot(s, e2)/dot( normal, ray_direction)
# v = dot(s, e1)/dot(-normal, ray_direction)
function hit(triangle::Triangle{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    vertices = triangle.vertices
    
    e1 = vertices[2] - vertices[1]
    e2 = vertices[3] - vertices[1]
    origin_v1 = vertices[1] - r.origin
    
    normal = cross(e1, e2)
    s = cross(origin_v1, r.direction) # s = cross(ray_direction, -origin_v1) = -cross(-origin_v1, ray_direction) = cross(origin_v1, ray_direction)
    c = inv(dot(normal, r.direction))
    
    u = dot(s, e2) * c
    v = dot(s, e1) * -c
    t = dot(normal, origin_v1) * c
    
    if u < zero(F) || v < zero(F) || u+v > one(F) || t < t_min || t_max < t
        return HitRecord(F, T)
    end
    
    p = at(r, t)
    front_face = dot(r.direction, normal) < zero(F)
    return HitRecord(t, p, (-1+(front_face<<1)) * normal, triangle.material, front_face, u, v)
end