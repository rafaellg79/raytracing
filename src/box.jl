include("aarect.jl")

struct Box{F, T} <: Hittable{T}
    box_min::Vec3{F}
    box_max::Vec3{F}
    sides::Tuple{xy_rect{F, T}, xy_rect{F, T}, xz_rect{F, T}, xz_rect{F, T}, yz_rect{F, T}, yz_rect{F, T}}
end

function Box(p0::Vec3{F}, p1::Vec3{F}, material::T) where {F, T}
    box_min = p0
    box_max = p1
    
    sides = (
        xy_rect{F, T}(p0.x, p1.x, p0.y, p1.y, p1.z, material), # Front
        xy_rect{F, T}(p0.x, p1.x, p0.y, p1.y, p0.z, material), # Back
        
        xz_rect{F, T}(p0.x, p1.x, p0.z, p1.z, p1.y, material), # Top
        xz_rect{F, T}(p0.x, p1.x, p0.z, p1.z, p0.y, material), # Bottom
        
        yz_rect{F, T}(p0.y, p1.y, p0.z, p1.z, p1.x, material), # Right
        yz_rect{F, T}(p0.y, p1.y, p0.z, p1.z, p0.x, material), # Left
    )
    return Box(box_min, box_max, sides)
end

function Box(p0::Vec3{F}, p1::Vec3{F}, materials::NTuple{6, T}) where {F, T<:Material}
    box_min = p0
    box_max = p1
    
    sides = (
        xy_rect{F, T}(p0.x, p1.x, p0.y, p1.y, p1.z, material[1]), # Front
        xy_rect{F, T}(p0.x, p1.x, p0.y, p1.y, p0.z, material[2]), # Back
        
        xz_rect{F, T}(p0.x, p1.x, p0.z, p1.z, p1.y, material[3]), # Top
        xz_rect{F, T}(p0.x, p1.x, p0.z, p1.z, p0.y, material[4]), # Bottom
        
        yz_rect{F, T}(p0.y, p1.y, p0.z, p1.z, p1.x, material[5]), # Right
        yz_rect{F, T}(p0.y, p1.y, p0.z, p1.z, p0.x, material[6]), # Left
    )
    return Box(box_min, box_max, sides)
end

function hit(box::Box{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F, T}
    closest_hit = HitRecord(F, T)
    
    h = hit(box.sides[1], r, t_min, t_max)
    if h.t < closest_hit.t
        closest_hit = h
        t_max = h.t
    end
    
    h = hit(box.sides[2], r, t_min, t_max)
    if h.t < closest_hit.t
        closest_hit = h
        t_max = h.t
    end
    
    h = hit(box.sides[3], r, t_min, t_max)
    if h.t < closest_hit.t
        closest_hit = h
        t_max = h.t
    end
    
    h = hit(box.sides[4], r, t_min, t_max)
    if h.t < closest_hit.t
        closest_hit = h
        t_max = h.t
    end
    
    h = hit(box.sides[5], r, t_min, t_max)
    if h.t < closest_hit.t
        closest_hit = h
        t_max = h.t
    end
    
    h = hit(box.sides[6], r, t_min, t_max)
    if h.t < closest_hit.t
        closest_hit = h
        t_max = h.t
    end
    
    return closest_hit
end

function bounding_box(box::Box)
    return AABB(box.box_min, box.box_max)
end