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