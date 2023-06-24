include("hittable.jl")
include("vertex.jl")

struct Triangle{V, T} <: Hittable{T}
    v1::V
    v2::V
    v3::V
    material::T
end

function bounding_box(t::Triangle{<:Vertex{F}}) where F<:AbstractFloat
    v1 = t.v1.position
    v2 = t.v2.position
    v3 = t.v3.position
    
    x_min = min(v1.x, v2.x, v3.x)-F(0.0001)
    y_min = min(v1.y, v2.y, v3.y)-F(0.0001)
    z_min = min(v1.z, v2.z, v3.z)-F(0.0001)
    x_max = max(v1.x, v2.x, v3.x)+F(0.0001)
    y_max = max(v1.y, v2.y, v3.y)+F(0.0001)
    z_max = max(v1.z, v2.z, v3.z)+F(0.0001)
    return AABB{F}(Vec3{F}(x_min, y_min, z_min), Vec3{F}(x_max, y_max, z_max))
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
# Let s = cross(ray_direction, ray_origin - V1) and c = 1/dot( normal, ray_direction)
# then substitute in the equations
# u = dot(s, e2) *  c
# v = dot(s, e1) * -c
#
# Doing the same to t
# t = dot(normal, V1 - ray_origin) * c
function hit(v1::Vec3{F}, v2::Vec3{F}, v3::Vec3{F}, r::Ray{F}, t_min::F, t_max::F) where F<:AbstractFloat
    e1 = v2 - v1
    e2 = v3 - v1
    origin_v1 = v1 - r.origin
    
    normal = cross(e1, e2)
    s = cross(origin_v1, r.direction) # s = cross(ray_direction, -origin_v1) = -cross(-origin_v1, ray_direction) = cross(origin_v1, ray_direction)
    c = inv(dot(normal, r.direction))
    
    u = dot(s, e2) *  c
    v = dot(s, e1) * -c
    t = dot(normal, origin_v1) * c
    
    if u < zero(F) || v < zero(F) || u+v > one(F) || t < t_min || t_max < t
        return F(Inf), normal, F(-1), F(-1)
    end
    
    return t, normal, u, v
end

hit(v1::V, v2::V, v3::V, r::Ray{F}, t_min::F, t_max::F) where V<:Vertex{F} where F = hit(v1.position, v2.position, v3.position, r, t_min, t_max)

function get_triangle_vertex_at_uv(triangle::Triangle, u::Real, v::Real)
    w = 1-u-v
    vertex = triangle.v1*w + triangle.v2*u + triangle.v3*v
    return vertex
end

function hit(triangle::Triangle{V, T}, r::Ray{F}, t_min::F, t_max::F) where {V, F<:AbstractFloat, T<:Material}
    v1, v2, v3 = triangle.v1, triangle.v2, triangle.v3
    t, normal, u, v = hit(v1, v2, v3, r, t_min, t_max)
    if t == F(Inf)
        return HitRecord(F, T)
    end
    
    vertex = get_triangle_vertex_at_uv(triangle, u, v)
    
    vertex_has_normal_data = almost_zero(vertex.normal)
    vertex_has_uv_data = vertex.u >= 0 && vertex.v >= 0
    
    normal = vertex_has_normal_data * vertex.normal + (!vertex_has_normal_data) * normal
    u = vertex_has_uv_data * vertex.u + !vertex_has_uv_data * u
    v = vertex_has_uv_data * vertex.v + !vertex_has_uv_data * v
    
    p = at(r, t)
    front_face = dot(r.direction, normal) < zero(F)
    
    return HitRecord(t, p, (-1+(front_face<<1)) * normalize(normal), triangle.material, front_face, u, v)
end

function cu_hit(v1::Vec3{F}, v2::Vec3{F}, v3::Vec3{F}, material::T, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    e1 = v2 - v1
    e2 = v3 - v1
    origin_v1 = v1 - r.origin
    
    normal = cross(e1, e2)
    s = cross(origin_v1, r.direction) # s = cross(ray_direction, -origin_v1) = -cross(-origin_v1, ray_direction) = cross(origin_v1, ray_direction)
    c = inv(dot(normal, r.direction))
    
    u = dot(s, e2) *  c
    v = dot(s, e1) * -c
    t = dot(normal, origin_v1) * c
    
    if u < zero(F) || v < zero(F) || u+v > one(F) || t < t_min || t_max < t
        return HitRecord(F, T)
    end
    
    p = at(r, t)
    front_face = dot(r.direction, normal) < zero(F)
    return HitRecord(t, p, (-1+(front_face<<1)) * normalize(normal), material, front_face, u, v)
end

cu_hit(triangle::Triangle{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material} = hit(triangle, r, t_min, t_max)#cu_hit(triangle.v1, triangle.v2, triangle.v3, triangle.material, r, t_min, t_max)