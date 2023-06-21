include("src/Ray_Tracing.jl")
include("scenes.jl")

using FileIO, MeshIO

"""
    render([F=Float32], world::Vector{<:Hittable}; kwargs...)

Ray trace `world` and returns an `Array{RGB{F}}` with the rendered image.

#Arguments:
- `width` : width in pixels 
- `height` : height in pixels
- `camera` : `Camera` object viewing the world
- `background` : a `Vec3` with the background color by default it's sky blue if no emissive object exists, otherwise it's black
- `pattern` : matrix `[x_0 ... x_n; y_0 ... y_n]` with the horizontal and vertical offsets of the ray direction
- `gpu` : set true to use CUDA.jl implementation

Note: the CUDA.jl implementation does not support all CPU features.
"""
function render(::Type{F}, world::Vector{<:Hittable};
                width=1280, 
                height=720, 
                camera=Camera(F), 
                background=any(obj->obj.material.type==Emissive, world) ? zero(Vec3{F}) : Vec3{F}(0.70, 0.80, 1.00), 
                pattern=F.(MSAA_S16), 
                gpu::Bool=false,
                kwargs...
               ) where F<:AbstractFloat
    world = SceneManager(world)
    
    img = zeros(RGB{F}, height, width)
    
    if gpu
        d_img = CUDA.zeros(eltype(img), height, width)
        
        draw!(d_img, world, camera; background = background, pattern = pattern)
        img = Array(d_img)
    else
        draw!(img, world, camera; background = background, pattern = pattern)
    end
    
    return img
end

render(world::Vector{<:Hittable}; kwargs...) = render(Float32, world; kwargs...)

"""
    render([F=Float32], filename::String, material;kwargs...)

Load a triangle mesh model from `filename` using `material` for the triangles material and ray trace the loaded mesh.
"""
function render(::Type{F}, filename::String, material::T; kwargs...) where {F<:AbstractFloat, T<:Material}
    world = Hittable[]
    
    mesh = load(filename)
    for tri in mesh
        push!(world, Triangle(Vec3{Float32}(tri[1]...), Vec3{Float32}(tri[2]...), Vec3{Float32}(tri[3]...), material))
    end
    render(world; kwargs...)
end

render(filename::String, material::Material; kwargs...) = render(Float32, filename, material; kwargs...)

"""
    render([F=Float32], key::Symbol; kwargs...)

Render a scene from `scene_list()` given the scene `key` with `kwargs...` parameters and store in `output_filename`.
"""
function render(::Type{F}, key::Symbol; gpu::Bool=false, kwargs...) where F<:AbstractFloat
    world, scene_kwargs = get_scene(key; F=F, kwargs...)
    kwargs = merge(scene_kwargs, kwargs)
    render(world; kwargs..., gpu=gpu)
end

render(key::Symbol; kwargs...) = render(Float32, key; kwargs...)

nothing
