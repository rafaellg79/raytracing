include("src/MSAA_patterns.jl")
scenes = Dict{Symbol, Function}()

for filename in readdir("samples")
    include(joinpath("samples", filename))
end

function get_scene(key::Symbol; kwargs...)
    width, height, camera, world, background, pattern = scenes[key](;kwargs...)
    return world, (width=width, height=height, camera=camera, background=background, pattern=pattern)
end
scenes_list() = keys(scenes)