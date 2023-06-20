# Ray Tracing in One Weekend

A Ray Tracer based on the [Ray Tracing in One Weekend Series](https://raytracing.github.io/) written in [Julia](https://julialang.org/).

![Cover from The Next Week rendered in Julia](/outputs/cover2_10000_samples.png)

This implementation supports GPU parallel processing with the [**CUDA.jl**](https://github.com/JuliaGPU/CUDA.jl) API.
However, not all features from the CPU implementation are supported in the **CUDA.jl** version.

In order to support several types of objects the GPU compiler requires all types to be plain data types (i.e. collection of primitive types).
To achieve that, heavy specialization of methods are performed by the JIT compiler. This implies in a slow compilation, but fast machine code.

Alternatively, multithreading is also supported by using the `--threads` command line argument when starting Julia.
For example:

```bash
$ julia --threads 4
```

Will start Julia with 4 threads.

All necessary packages are included in the Project.toml file.
To install the necessary packages from the Julia REPL, enter the Pkg mode (type `]`) and run:
```julia
pkg> activate <path_to_Project.toml>
pkg> instantiate
```

## Example Usage

To render sample scenes first include the main.jl file and call the `render` method with one of the available scenes from `scenes_list()`.

For example, the code below looks at the scenes available and render the `:cornell_box` scene:

```julia
julia> include("main.jl")

julia> scenes_list()
KeySet for a Dict{Symbol, Function} with 5 entries. Keys:
  :cornell_box
  :cover1
  :cu_cover2
  :cover2
  :cornell_smoke_box

julia> img = render(:cornell_box));
400/400
```

After ray tracing the scene, the resulting image is returned as an `Array{RGB}` which can be used with most image IO packages to store the image.
The **Images.jl** package included in the Project.toml will already have one installed so you can just call the `save` method to save the image:

```julia
julia> save(joinpath("outputs", "cornell_box_1000_samples.png"), img)
┌ Warning: Mapping to the storage type failed; perhaps your data had out-of-range values?
│ Try `map(clamp01nan, img)` to clamp values to a valid range.
└ @ ImageMagick ~/.julia/packages/ImageMagick/b8swT/src/ImageMagick.jl:180
310007
```

As the `Warning` explains some values are *out-of-range*, which is not supported by the **ImageMagick.jl** backend used to save the image.
This happens because the scene is based on the Cornell box code from the Next Week book which has a light with value 15 (see [listing 57](https://raytracing.github.io/books/RayTracingTheNextWeek.html#rectanglesandlights/creatinganempty%E2%80%9Ccornellbox%E2%80%9D)).
To solve this just do as suggested and clamp the image values to [0, 1]:

```julia
julia> save(joinpath("outputs", "cornell_box_1000_samples.png"), clamp01.(img))
```

The saved image is displayed below.

![Cornell box rendered with 1000 samples per pixel](/outputs/cornell_box_1000_samples.png)

By default the scene uses 1000 *samples per pixel* (spp) and contains a lot of noise due to rays bouncing out of the scene.
We can increase the number of samples per pixel to reduce the noise.

Let's increase the number of samples to 10000 and use the gpu to process the image faster with the `spp` and `gpu` keyword arguments.

```
julia> img = render(:cornell_box; spp=10000, gpu=true);
100.000000%
julia> save(joinpath("outputs", "cornell_box_10000_samples.png"), clamp01.(img))
```

Using `spp=10000` reduces a lot of the noise as shown below.

![Cornell box rendered with 10000 samples per pixel](/outputs/cornell_box_10000_samples.png)
