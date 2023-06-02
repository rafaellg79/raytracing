include("vec3.jl")

struct Camera{F}
    origin::Vec3{F}
    lower_left_corner::Vec3{F}
    horizontal::Vec3{F}
    vertical::Vec3{F}
    u::Vec3{F}
    v::Vec3{F}
    w::Vec3{F}
    lens_radius::F
    t0::F
    t1::F
end

function Camera(origin::Vec3{F}, direction::Vec3{F}, up::Vec3{F}, fov::F, aspect_ratio::F, aperture::F, focus_dist::F, t0::F, t1::F) where F<:AbstractFloat
    theta = deg2rad(fov)
    l = tan(theta / 2)
    viewport_width = 2l
    viewport_height = viewport_width / aspect_ratio
    
    w = normalize(direction)
    u = normalize(cross(up, w))
    v = cross(w, u)
    
    horizontal = focus_dist * viewport_width * u
    vertical = focus_dist * viewport_height * v
    lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w
    return Camera{F}(origin, lower_left_corner, horizontal, vertical, u, v, w, aperture/2, t0, t1)
end

Camera(origin::Vec3{F}, direction::Vec3{F}, up::Vec3{F}, fov::F, aspect_ratio::F, aperture::F, focus_dist::F) where F<:AbstractFloat = Camera(origin, direction, up, fov, aspect_ratio, aperture, focus_dist, zero(F), zero(F))

function get_ray(camera::Camera{F}, s::F, t::F) where F <: AbstractFloat
    rd = camera.lens_radius * random_in_unit_disk(Vec3{F})
    offset_origin = camera.u * rd.x + camera.v * rd.y + camera.origin
    return Ray(offset_origin, camera.lower_left_corner + s*camera.horizontal + t*camera.vertical - offset_origin, rand(F) * (camera.t1 - camera.t0) + camera.t0)
end