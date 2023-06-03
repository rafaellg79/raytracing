include("cuBVH.jl")
include("scene_manager.jl")

# CUDA version of SceneManager
struct CuSceneManager{T}
    scene::T
end

import Adapt

function Adapt.adapt_structure(to, cu_scene::CuSceneManager)
    scene = Adapt.adapt_structure.(Ref(to), cu_scene.scene)
    CuSceneManager{typeof(scene)}(scene)
end

# cu_type functions
# Receive a type and return the CUDA equivalent type recursively 
# substituting the parametric types by CUDA equivalent types
# For example cu_type(Transform{BVHTree}) == Transform{CuBVHTree}
function cu_type(::Type{T}) where T
    return T
end

function cu_type(::Type{ConstantMedium{T}}) where T
    return ConstantMedium{cu_type(T)}
end

function cu_type(::Type{Transform{T}}) where T
    return Transform{cu_type(T)}
end

function cu_type(::Type{Vector{T}}) where T
    return CuVector{cu_type(T)}
end

function cu_type(::Type{BVHTree{T}}) where T
    return CuBVHTree{CuVector{BVHNode, CUDA.Mem.DeviceBuffer}, CuVector{cu_type(T), CUDA.Mem.DeviceBuffer}}
end

# cu function overload for parametric types
function cu(transforms_h::Vector{Transform{BVHTree{T}}}) where T
    cu_transforms_h = [cu(transform) for transform in transforms_h]
    return cu(cu_transforms_h)
end

function cu(transforms_h::Vector{Transform{Vector}})
    cu_transforms_h = [cu(transform) for transform in transforms_h]
    return cu(cu_transforms_h)
end

function cu(trees_h::Vector{BVHTree})
    cu_trees_h = [cu(tree) for tree in transforms_h]
    return cu(cu_trees_h)
end

function cu(constant_medium::ConstantMedium{T}) where T
    return ConstantMedium(cu(constant_medium.boundary), constant_medium.phase_function, constant_medium.neg_inv_density)
end

function cu(transform::Transform{T}) where T
    return Transform(cu(transform.object), transform.xform, transform.inv_xform, transform.bbox)
end

function CuSceneManager(scene::SceneManager)
    d_scene = cu.(scene)
    return CuSceneManager{typeof(d_scene)}(d_scene)
end

function cu_hit(buffer::CuDeviceArray, objects::CuDeviceArray{T}, r::Ray{F}, t_min::F, t_max::F) where {T<:Hittable, F<:AbstractFloat}
    closest_h = HitRecord()
    closest_t = t_max
    for i in 1:length(objects)
        h = cu_hit(buffer, objects[i], r, t_min, closest_t)
        if h.t < closest_h.t
            closest_h = h
            closest_t = h.t
        end
    end
    return closest_h
end

function cu_hit(buffer::CuDeviceArray, transform::Transform{T}, r::Ray{F}, t_min::F, t_max::F) where {F <: AbstractFloat, T <:Hittable}
    hit_result = cu_hit(buffer, transform.object, Ray(apply_point(transform.inv_xform, r.origin), apply_vec(transform.inv_xform, r.direction), r.t), t_min, t_max)
    
    if hit_result.t == Inf
        return hit_result
    end
    
    t = hit_result.t
    p = at(r, t)
    normal = apply_vec(transform.xform, hit_result.normal)
    
    return HitRecord(t, p, normal, hit_result.material, hit_result.front_face, hit_result.u, hit_result.v)
end

function cu_hit(buffer::CuDeviceArray, scene_manager::CuSceneManager, ray::Ray{F}, t_min::F, t_max::F) where F<:AbstractFloat
    return mapreduce(s -> cu_hit(buffer, s, ray, t_min, t_max), (a, b) -> a.t <= b.t ? a : b, scene_manager.scene)
end

cu_hit(buffer::CuDeviceArray, scene_manager::CuSceneManager, ray::Ray{F}) where F = cu_hit(buffer, scene_manager, ray, F(0.001), F(Inf))