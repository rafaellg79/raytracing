include("MSAA_patterns.jl")
scenes = Dict{Symbol, Function}()

for filename in readdir("samples")
    if filename[1] == 'c' || filename[1] == 'C'
        include(joinpath("samples", filename))
    end
end

get_scene(::Type{F}, key::Symbol, args...) where F<:AbstractFloat = scenes[key](F, args...)
get_scene(key::Symbol, args...) = scenes[key](args...)
scenes_list() = keys(scenes)