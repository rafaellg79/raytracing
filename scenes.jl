include("src/MSAA_patterns.jl")
scenes = Dict{Symbol, Function}()

for filename in readdir("samples")
    include(joinpath("samples", filename))
end

get_scene(::Type{F}, key::Symbol, args...) where F<:AbstractFloat = scenes[key](F, args...)
get_scene(key::Symbol, args...) = scenes[key](args...)
scenes_list() = keys(scenes)