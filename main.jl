include("src/Ray_Tracing.jl")
include("scenes.jl")

"""
    render([F=Float32], world::Vector{<:Hittable}; kwargs...)

Ray trace `world` and saves the resulting image into a file.

#Arguments:
- `width` : width in pixels 
- `height` : height in pixels
- `camera` : `Camera` object viewing the world
- `background` : a `Vec3` with the background color by default it's sky blue if no emissive object exists, otherwise it's black
- `pattern` : matrix `[x_0 ... x_n; y_0 ... y_n]` with the horizontal and vertical offsets of the ray direction
- `output_filename` : file in which the resulting image will be stored with format inferred by file extension (see [`save`](@ref) for supported formats).
- `gpu` : set true to use CUDA.jl implementation

Note: the CUDA.jl implementation does not support all CPU features.
"""
function render(::Type{F}, world::Vector{<:Hittable};
                width=1280, 
                height=720, 
                camera=Camera(F), 
                background=any(obj->obj.material.type==Emissive, world) ? zero(Vec3{F}) : Vec3{F}(0.70, 0.80, 1.00), 
                pattern=F.(MSAA_S16), 
                gpu::Bool=false) where F<:AbstractFloat
    world = SceneManager(world)
    
    img = zeros(RGB{F}, height, width)
    
    if gpu
        d_img = CUDA.zeros(eltype(img), height, width)
        
        draw!(d_img, world, camera; background = background, pattern = pattern)
        img = Array(d_img)
    else
        draw!(img, world, camera; background = background, pattern = pattern)
    end
    
    save(output_filename, clamp01nan.(img))
    display("Saved image in $output_filename")
end

render(world::Vector{<:Hittable}; kwargs...) = render(Float32, world; kwargs...)

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