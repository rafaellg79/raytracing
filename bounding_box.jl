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