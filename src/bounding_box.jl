include("vec3.jl")

struct AABB{F}
    min::Vec3{F}
    max::Vec3{F}
end

function surrounding_box(boxes::AbstractVector{AABB{F}}) where F<:AbstractFloat
    global_min = Vec3{F}(minimum(box.min.x for box in boxes), minimum(box.min.y for box in boxes), minimum(box.min.z for box in boxes))
    global_max = Vec3{F}(maximum(box.max.x for box in boxes), maximum(box.max.y for box in boxes), maximum(box.max.z for box in boxes))
    return AABB{F}(global_min, global_max)
end

function inside(p::Vec3{F}, box::AABB{F}) where F<:AbstractFloat
    return box.min <= p && p <= box.max
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