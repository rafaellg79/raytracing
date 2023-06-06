include("BVH.jl")
using CUDA

struct CuBVHTree{A, B} <: Hittable{B}
    nodes::A
    objects::B
end

textype(::Type{CuBVHTree{A, B}}) where {A, B} = textype(B)
textype(::Type{CuVector{T}}) where T = textype(T)

import Adapt, CUDA.cu

function Adapt.adapt_structure(to, bvh::CuBVHTree{A, B}) where {A, B}
    nodes = Adapt.adapt_structure(to, bvh.nodes)
    objects = Adapt.adapt_structure(to, bvh.objects)
    CuBVHTree(nodes, objects)
end

function cu(bvh::BVHTree{A}) where A
    return CuBVHTree(cu(bvh.nodes), cu(bvh.objects))
end

function cu_hit(buffer::CuDeviceArray, obj::T, ray::Ray{F}, t_min::F, t_max::F) where {F, T}
    return hit(obj, ray, t_min, t_max)
end

function cu_hit(obj::T, ray::Ray{F}, t_min::F, t_max::F) where {F, T}
    return hit(obj, ray, t_min, t_max)
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

function cu_hit(next::CuDeviceArray, tree::CuBVHTree{A, B}, ray::Ray{F}, t_min::F, t_max::F) where {A, B, F}
    tex = textype(B)
    
    if isempty(tree.nodes)
        return HitRecord(F, tex)
    end
    
    block_id = blockIdx()[3]-1 + ((blockIdx()[2]-1) + (blockIdx()[1]-1) * gridDim()[2]) * gridDim()[3]
    thread_buffer_id = threadIdx()[3] + ((threadIdx()[2]-1) + (threadIdx()[1]-1 + block_id * blockDim()[1]) * blockDim()[2]) * blockDim()[3]
    next[1, thread_buffer_id] = 1
    closest_hit = HitRecord(F, tex)
    h = HitRecord(F, tex)
    n = 1
    while n > 0
        node = tree.nodes[next[n, thread_buffer_id]]
        n -= 1
        
        if !cu_hit(node.box, ray, t_min, t_max)
            continue
        end
        
        if node.is_leaf
            h = cu_hit(tree.objects[node.left], ray, t_min, t_max)
            if h.t < closest_hit.t
                t_max = h.t
                closest_hit = h
            end
        else
            n += 1
            next[n, thread_buffer_id] = node.left
            n += 1
            next[n, thread_buffer_id] = node.right
        end
    end
    return closest_hit
end