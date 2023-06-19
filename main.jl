include("src/Ray_Tracing.jl")
include("scenes.jl")

# Use main with get_scene to render predefined scenes from the Ray Tracing in One Weekend series.
function main(width, height, camera, world, background, pattern; output_filename=joinpath("outputs", "image.png"), gpu::Bool=false)
    world = SceneManager(world)
    
    img = zeros(RGB{eltype(pattern)}, height, width)
    
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

main(key::Symbol; output_filename=joinpath("outputs", "image.png"), gpu::Bool=false, args...) = main(get_scene(key, args...)...; output_filename=output_filename, gpu=gpu)