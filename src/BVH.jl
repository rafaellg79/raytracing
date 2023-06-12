include("Hittable.jl")

# If node::BVHNode is an internal node then node.is_leaf == false and node.left and node.right are the indices of it's children in tree.nodes, i.e. tree.nodes[node.left] and tree.nodes[node.right]
# If node::BVHNode is a leaf node then node.is_leaf == true and tree.objects[node.left:node.right] are the objects inside the node
struct BVHNode{F}
    box::AABB{F}
    left::Int
    right::Int
    is_leaf::Bool
end

struct BVHTree{F, T<:Hittable} <: Hittable{T}
    nodes::Vector{BVHNode{F}}
    objects::Vector{T}
    traversal_buffer::Vector{Int}
end

import Base.eltype

eltype(::Type{BVHTree{F, T}}) where {F, T} = T

function hit(tree::BVHTree{F, T}, ray::Ray{F}, t_min::F, t_max::F) where {F <: AbstractFloat, T <: Hittable}
    if isempty(tree.nodes)
        return HitRecord(F, textype(T))
    end
    next = tree.traversal_buffer
    n = 1
    next[1] = 1
    closest_hit = HitRecord(F, textype(T))
    h = HitRecord(F, textype(T))
    while n > 0
        node = tree.nodes[next[n]]
        n -= 1
        
        if !hit(node.box, ray, t_min, t_max)
            continue
        end
        
        if node.is_leaf
            h = hit((@view tree.objects[node.left:node.right]), ray, t_min, t_max)
            if h.t < closest_hit.t
                closest_hit = h
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

hit(tree::BVHTree{F, T}, ray::Ray{F}) where {F, T} = hit(tree, ray, F(0.001), F(Inf))

function bounding_box(bvh::BVHNode)
    return bvh.box
end

function bounding_box(bvh::BVHTree{T}) where T
    return bvh.nodes[1].box
end

function build_bvh!(tree::BVHTree{F}; node_id::Int=1, left::Int=1, right::Int=length(tree.objects)) where F<:AbstractFloat
    @assert left <= right "left($left) > right($right) on build_bvh! recursion"
    objects = tree.objects
    if left == right
        tree.nodes[node_id] = BVHNode{F}(surrounding_box(@view tree.objects[left:right]), left, right, true)
        return node_id
    end
    
    # Compute the bounding box of all objects
    box = surrounding_box(@view objects[left:right])
    
    axis_length = box.max - box.min
    axis = findmax(getfield(axis_length, i) for i in 1:3)[2]
    if axis == 1
        comparator = (a, b) -> (bounding_box(a).min.x < bounding_box(b).min.x)
    elseif axis == 2
        comparator = (a, b) -> (bounding_box(a).min.y < bounding_box(b).min.y)
    else
        comparator = (a, b) -> (bounding_box(a).min.z < bounding_box(b).min.z)
    end
    sort!((@view objects[left:right]); lt=comparator)
    
    # Recursively visit the left and right children
    mid = (left+right) ÷ 2
    left_node = build_bvh!(tree; node_id = node_id+1, left=left, right=mid)
    right_node = build_bvh!(tree; node_id = node_id + 2(mid + 1 - left), left=mid+1, right=right)
    
    # Create a new BVH node
    tree.nodes[node_id] = BVHNode{F}(box, left_node, right_node, false)
    
    return node_id
end

function build_bvh(objects::Vector{<:Hittable})
    if isempty(objects)
        return BVHTree{Float32, eltype(objects)}(BVHNode{Float32}[], objects, Int[])
    end
    # TODO: bit of a hack to determine the appropriate floating point type, should change this later
    F = typeof(textype(typeof(objects[1]))().fuzz)
    nodes = Vector{BVHNode{F}}(undef, 2*length(objects)-1)
    traversal_buffer = Vector{Int}(undef, ceil(Int, log2(length(nodes)))+1)
    tree = BVHTree(nodes, objects, traversal_buffer)# In the worst case the tree is full, so we have 2*num_leaves - 1 nodes
    build_bvh!(tree)
    return tree
end