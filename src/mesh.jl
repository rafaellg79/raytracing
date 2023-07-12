include("triangle.jl")
include("hittable.jl")
include("cuBVH.jl")

struct Mesh{Tri, Tex, I} <: Hittable{Tex}
    vertices::Tri
    materials::Tex
    indices::I
end

Mesh(vertices::Tri, materials::Tex) where {Tri, Tex<:Material} = Mesh{Tri, Tex, Vector{Int}}(vertices, materials, [reshape(collect(1:size(vertices, 2)), 3, :); collect(1:length(materials))'])

textype(::Type{Mesh{Tri, Tex, I}}) where {Tri, Tex, I} = textype(Tex)

import Adapt

function Adapt.adapt_structure(to, mesh::Mesh)
    vertices = Adapt.adapt_structure(to, mesh.vertices)
    materials = Adapt.adapt_structure(to, mesh.materials)
    indices = Adapt.adapt_structure(to, mesh.indices)
    Mesh(vertices, materials, indices)
end

function Adapt.adapt_structure(to, bvh::CuBVHTree{A, B}) where {A, B<:AbstractArray{<:Mesh}}
    nodes = Adapt.adapt_structure(to, bvh.nodes)
    
    # Adapt.jl doesn't perform wrapping recursively 
    # So we must copy the Mesh data from GPU to CPU
    host_objects = Array(bvh.objects)
    
    # Adapt each Mesh on host
    host_objects = map(obj -> Adapt.adapt_structure(to, obj), host_objects)
    
    # Create a GPU array with the adapted type
    device_objects = similar(bvh.objects, eltype(host_objects))
    
    # Copy host data back to GPU
    copyto!(device_objects, host_objects)
    
    # Finally perform adapt on the array
    device_objects = Adapt.adapt_structure(to, device_objects)
    CuBVHTree(nodes, device_objects)
end 

function bounding_box(mesh::Mesh{Tri}; indices=1:length(mesh.vertices)) where Tri<:AbstractVector{<:Vertex}
    min = minimum(v -> v.position, (@view mesh.vertices[indices])) - 0.0001
    max = maximum(v -> v.position, (@view mesh.vertices[indices])) + 0.0001
    return AABB(min, max)
end

import Base.getindex

function getindex(mesh::Mesh, ind::Integer)
    v1 = mesh.vertices[mesh.indices[1, ind]]
    v2 = mesh.vertices[mesh.indices[2, ind]]
    v3 = mesh.vertices[mesh.indices[3, ind]]
    material = mesh.materials[mesh.indices[4, ind]]
    return Triangle(v1, v2, v3, material)
end

function hit(mesh::Mesh{<:AbstractVector, <:AbstractVector{T}}, inds, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    closest_hit = HitRecord(F, T)
    for i in inds
        h = hit(mesh[i], r, t_min, t_max)
        if h.t < closest_hit.t
            closest_hit = h
            t_max = closest_hit.t
        end
    end
    return closest_hit
end

function hit(mesh::Mesh{<:AbstractVector, <:AbstractVector{T}}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    if isempty(mesh.vertices)
        return HitRecord(F, T)
    end
    closest_hit = HitRecord(F, T)
    for i in 1:size(mesh.indices, 2)
        h = hit(mesh[i], r, t_min, t_max)
        if h.t < closest_hit.t
            closest_hit = h
            t_max = closest_hit.t
        end
    end
    return closest_hit
end

function cu_hit(mesh::Mesh{<:AbstractVector, <:AbstractVector{T}}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    if isempty(mesh.vertices)
        return HitRecord(F, T)
    end
    closest_hit = HitRecord(F, T)
    for i in 1:size(mesh.indices, 2)
        h = cu_hit(mesh[i], r, t_min, t_max)
        if h.t < closest_hit.t
            closest_hit = h
            t_max = closest_hit.t
        end
    end
    return closest_hit
end

function cu_hit(mesh::Mesh{<:AbstractVector, <:AbstractVector{T}}, inds, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
    closest_hit = HitRecord(F, T)
    for i in inds
        h = cu_hit(mesh[i], r, t_min, t_max)
        if h.t < closest_hit.t
            closest_hit = h
            t_max = closest_hit.t
        end
    end
    return closest_hit
end

function hit(tree::BVHTree{F, T}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Mesh}
    tex = textype(T)
    if isempty(tree.nodes)
        return HitRecord(F, tex)
    end
    
    next = @view tree.traversal_buffer[:, Threads.threadid()]
    n = 1
    next[1] = 1
    
    active_mesh_index = 0
    closest_hit = HitRecord(F, tex)
    h = HitRecord(F, tex)
    while n > 0
        if active_mesh_index == n
            active_mesh_index = 0
            n -= 1
            continue
        end
        node = tree.nodes[next[n]]
        n -= 1
        
        if !hit(node.box, r, t_min, t_max)
            continue
        end
        
        if node.is_leaf
            if active_mesh_index == 0
                n += 1
                next[n] = node.right
                active_mesh_index = n
                
                n += 1
                next[n] = node.left
            else
                h = hit(tree.objects[next[active_mesh_index]], node.left:node.right, r, t_min, t_max)
                if h.t < closest_hit.t
                    closest_hit = h
                end
            end
        else
            n += 1
            next[n] = node.left
            n += 1
            next[n] = node.right
        end
    end
    return closest_hit
end

function cu_hit(next::CuDeviceArray, tree::CuBVHTree{A, <:CuDeviceArray{T}}, r::Ray{F}, t_min::F, t_max::F) where {A, T<:Mesh, F}
    tex = textype(T)
    if isempty(tree.nodes)
        return HitRecord(F, tex)
    end
    
    block_id = blockIdx()[3]-1 + ((blockIdx()[2]-1) + (blockIdx()[1]-1) * gridDim()[2]) * gridDim()[3]
    thread_buffer_id = threadIdx()[3] + ((threadIdx()[2]-1) + (threadIdx()[1]-1 + block_id * blockDim()[1]) * blockDim()[2]) * blockDim()[3]
    next[1, thread_buffer_id] = 1
    n = 1
    
    active_mesh_index = 0
    closest_hit = HitRecord(F, tex)
    h = HitRecord(F, tex)
    while n > 0
        if active_mesh_index == n
            active_mesh_index = 0
            n -= 1
            continue
        end
        node = tree.nodes[next[n, thread_buffer_id]]
        n -= 1
        
        if !cu_hit(node.box, r, t_min, t_max)
            continue
        end
        
        if node.is_leaf
            if active_mesh_index == 0
                n += 1
                next[n, thread_buffer_id] = node.right
                active_mesh_index = n
                
                n += 1
                next[n, thread_buffer_id] = node.left
            else
                h = cu_hit(tree.objects[next[active_mesh_index, thread_buffer_id]], node.left:node.right, r, t_min, t_max)
                if h.t < closest_hit.t
                    t_max = h.t
                    closest_hit = h
                end
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

function build_bvh!(tree::BVHTree{F, T}, mesh::T; node_id::Int=1, left::Int=1, right::Int=size(mesh.indices, 2), max_triangles=1, left_child=false) where {F<:AbstractFloat, T<:Mesh}
    @assert left <= right "left($left) > right($right) on build_bvh! recursion"
    if right - left < max_triangles
        tree.nodes[node_id] = BVHNode{F}(bounding_box(mesh; indices=(@view mesh.indices[1:3, left:right])), left, right, true)
        return node_id
    end
    
    # Compute the bounding box of all objects
    box = bounding_box(mesh; indices=(@view mesh.indices[1:3, left:right]))
    
    axis_length = box.max - box.min
    axis = findmax(getfield(axis_length, i) for i in 1:3)[2]
    get_bounding_box = (indices) -> bounding_box(mesh; indices=(@view indices[1:3]))
    if axis == 1
        comparator = (a, b) -> (a.min.x < b.min.x)
    elseif axis == 2
        comparator = (a, b) -> (a.min.y < b.min.y)
    else
        comparator = (a, b) -> (a.min.z < b.min.z)
    end
    mesh.indices[:, left:right] .= sortslices((@view mesh.indices[:, left:right]); dims=2, by=get_bounding_box, lt=comparator)
    
    # Recursively visit the left and right children
    mid = (left+right) ÷ 2
    left_node = build_bvh!(tree, mesh; node_id = node_id * 2, left=left, right=mid, max_triangles=max_triangles, left_child=left_child)
    right_node = build_bvh!(tree, mesh; node_id = node_id * (2 - left_child) + 1, left=mid+1, right=right, max_triangles=max_triangles, left_child=false)
    
    # Create a new BVH node
    tree.nodes[node_id] = BVHNode{F}(box, left_node, right_node, false)
    
    return node_id
end

function build_bvh!(tree::BVHTree{F, T}; node_id::Int=1, left::Int=1, right::Int=length(tree.objects), max_triangles=1) where {F<:AbstractFloat, T<:Mesh}
    @assert left <= right "left($left) > right($right) on build_bvh! recursion"
    objects = tree.objects
    if left == right
        mesh = tree.objects[right]
        left_node = build_bvh!(tree, mesh; node_id = node_id * 2, max_triangles=max_triangles, left_child=true)
        tree.nodes[node_id] = BVHNode{F}(bounding_box(tree.objects[left]), left_node, right, true)
        return node_id
    end
    
    # Compute the bounding box of all objects
    box = surrounding_box(@view objects[left:right])
    
    axis_length = box.max - box.min
    axis = findmax(getfield(axis_length, i) for i in 1:3)[2]
    if axis == 1
        comparator = (a, b) -> (a.min.x < b.min.x)
    elseif axis == 2
        comparator = (a, b) -> (a.min.y < b.min.y)
    else
        comparator = (a, b) -> (a.min.z < b.min.z)
    end
    sort!((@view objects[left:right]); by=bounding_box, lt=comparator)
    
    # Recursively visit the left and right children
    mid = (left+right) ÷ 2
    left_node = build_bvh!(tree; node_id = node_id * 2, left=left, right=mid, max_triangles=max_triangles)
    right_node = build_bvh!(tree; node_id = node_id * 2 + 1, left=mid+1, right=right, max_triangles=max_triangles)
    
    # Create a new BVH node
    tree.nodes[node_id] = BVHNode{F}(box, left_node, right_node, false)
    
    return node_id
end

function build_bvh(mesh::T; max_triangles=1) where T<:Mesh
    F = eltype(mesh.vertices[1])
    nodes = Vector{BVHNode{F}}(undef, 2*size(mesh.indices, 2)-1)
    traversal_buffer = Vector{Int}(undef, ceil(Int, log2(length(nodes)))+1)
    tree = BVHTree(nodes, [mesh], traversal_buffer)
    build_bvh!(tree; max_triangles=max_triangles)
    return tree
end

function build_bvh(meshes::Vector{T}; max_triangles=1) where T<:Mesh
    if isempty(meshes)
        return BVHTree{Float32, Mesh}(BVHNode{Float32}[], meshes, Int[])
    end
    F = eltype(meshes[1].vertices[1])
    num_triangles = (mesh) -> size(mesh.indices, 2)
    nodes = Vector{BVHNode{F}}(undef, 4*sum(num_triangles, meshes)-1)
    traversal_buffer = Matrix{Int}(undef, ceil(Int, log2(length(nodes)))+1, Threads.nthreads())
    tree = BVHTree(nodes, meshes, traversal_buffer)
    build_bvh!(tree; max_triangles=max_triangles)
    return tree
end