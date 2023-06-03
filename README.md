# raytracing

A Ray Tracer based on the [Ray Tracing in One Weekend Series](https://raytracing.github.io/) written in [Julia](https://julialang.org/).

![Cover from The Next Week](/outputs/cover2_10000_samples.png)

This implementation supports GPU parallel processing with the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) API.
All features from the CPU implementation are supported in the CUDA.jl version with exception of textures.

In order to support several types of objects the GPU compiler requires all types to be plain data types (i.e. collection of primitive types).
To achieve that, heavy specialization of methods are performed by the JIT compiler. This implies in a slow compilation, but fast machine code.

All necessary packages are included in the Project.toml file.
To install the necessary packages from the Julia REPL, enter the Pkg mode (type `]`) and run:
```julia
pkg> activate .
pkg> instantiate
```

To render a few sample scenes call the `main` method with one of the available scenes from `scenes_list()`.

For example, the code below looks at the scenes available and render the `:cornell_box` scene:

```julia
julia> scenes_list()
KeySet for a Dict{Symbol, Function} with 5 entries. Keys:
  :cornell_box
  :cover1
  :cu_cover2
  :cover2
  :cornell_smoke_box

julia> main(:cornell_box; N=10000))
399/400
"Saved image in outputs/image.png"
```

The resulting image shown below is then saved in `outputs/image.png`.

![Cornell box rendered with 10000 samples per pixel](/outputs/cornell_box_10000_samples.png)
