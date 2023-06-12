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

function bounding_box(mesh::Mesh{Tri}) where Tri<:AbstractMatrix{F} where F<:AbstractFloat
    min = minimum(mesh.vertices; dims=2)
    max = maximum(mesh.vertices; dims=2)
    return AABB(Vec3{F}(min[1], min[2], min[3]), Vec3{F}(max[1], max[2], max[3]))
end

function bounding_box(mesh::Mesh{Tri}) where Tri<:AbstractVector{<:Vec3}
    min = reduce(Base.min, mesh.vertices) - 0.0001
    max = reduce(Base.max, mesh.vertices) + 0.0001
    return AABB(min, max)
end

function get_vertex(mesh::Mesh{<:AbstractVector{Vec3{F}}}, ind::Integer) where F<:AbstractFloat
    return mesh.vertices[ind]
end

function get_vertex(mesh::Mesh{<:AbstractMatrix{F}}, ind::Integer) where F<:AbstractFloat
    return Vec3{F}(mesh.vertices[1, ind], mesh.vertices[2, ind], mesh.vertices[3, ind])
end

import Base.getindex

function getindex(mesh::Mesh, ind::Integer)
    v1 = get_vertex(mesh, mesh.indices[1, ind])
    v2 = get_vertex(mesh, mesh.indices[2, ind])
    v3 = get_vertex(mesh, mesh.indices[3, ind])
    material = mesh.materials[mesh.indices[4, ind]]
    return Triangle(v1, v2, v3, material)
end

function hit(mesh::Mesh{<:AbstractVecOrMat, <:AbstractVector{T}}, r::Ray{F}, t_min::F, t_max::F) where {F<:AbstractFloat, T<:Material}
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