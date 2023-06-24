include("triangle.jl")
include("hittable.jl")

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