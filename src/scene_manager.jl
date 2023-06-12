include("hittable.jl")
include("BVH.jl")

const SceneManager = Tuple

textype(::Type{SceneManager{T}}) where T = textype(T)
textype(scene_manager::SceneManager) = textype(typeof(first(scene_manager)))

function SceneManager(objects::Vector{T}) where T<:Hittable
    fields = Tuple([obj_type[] for obj_type in unique!(typeof.(objects))])
    
    for obj in objects, field in fields
        if obj isa eltype(field)
            push!(field, obj)
        end
    end
    
    return fields
end

function SceneManager(object::H) where H<:Hittable
    return (object,)
end

function build_bvh_scene(objects::Vector{T}) where T<:Hittable
    if length(objects) == 0
        return BVHTree{T}(Vector{BVHNode}(undef, 0), Vector{T}(undef, 0), Vector{Int}(undef, 0))
    end
    return build_bvh(objects)
end

function build_bvh_scene(object::T) where T<:Hittable
    return object
end

function build_bvh(scene_manager::SceneManager)
    return map(build_bvh_scene, scene_manager)
end

function hit(scene_manager::SceneManager, ray::Ray{F}, t_min::F, t_max::F) where F
    closest_t = [t_max]
    closest_hit = (a, b) -> begin
        if a.t <= b.t
            a
        else
            closest_t[1] = b.t
            b
        end
    end
    return mapreduce(s -> hit(s, ray, t_min, closest_t[1]), closest_hit, scene_manager)
end

hit(scene_manager::SceneManager, ray::Ray{F}) where F<:AbstractFloat = hit(scene_manager, ray, F(0.001), F(Inf))