include("vec3.jl")
include("ray.jl")

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

# CPU version with early exits
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

# From Ray Tracing Gems II - Chapter 2
function cu_hit(h::AABB, r::Ray{F}, t_min::F, t_max::F) where F
    # Test for intersection with an AABB
    inv_ray_dir = one(F) / r.direction
    # Absolute distances to lower and upper box coordinates
    t_lower = (h.min - r.origin)*inv_ray_dir
    t_upper = (h.max - r.origin)*inv_ray_dir
    # The three t-intervals (for x-/y-/z-slabs , and ray p(t))
    t_mins = min(t_lower, t_upper)
    t_maxes = max(t_lower, t_upper)
    # Easy to remember: ``max of mins , and min of maxes ''
    t_box_min = max(t_mins.x, t_mins.y, t_mins.z, t_min)
    t_box_max = min(t_maxes.x, t_maxes.y, t_maxes.z, t_max)
    
    return t_box_min <= t_box_max
end