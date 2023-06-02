using Images, Random, CUDA
# Hittables
include("cuBVH.jl")
include("box.jl")
include("transform.jl")
include("constant_medium.jl")
include("scene_manager.jl")
include("cuscene_manager.jl")

# Utilities
include("camera.jl")

function path_trace(ray::Ray{F}, scene::SceneManager; background::Vec3{F}=zero(Vec3{F}), depth::Int=50) where F<:AbstractFloat
    accumulated_attenuation = Vec3{F}(1)
    color = zero(Vec3{F})
    while depth > 0
        hit_result = hit(scene, ray)
        if hit_result.t == F(Inf)
            return color + accumulated_attenuation * background
        end
        
        ray, attenuation = scatter(ray, hit_result)
        color += accumulated_attenuation * emit(hit_result)
        if almost_zero(attenuation)
            return color
        end
        accumulated_attenuation *= attenuation
        depth -= 1
    end
    return color
end

function path_trace(ray::Ray{F}, scene::Vector{T}; background::Vec3{F}=zero(Vec3{F}), depth::Int=50) where {F<:AbstractFloat, T<:Hittable}
    accumulated_attenuation = Vec3{F}(1)
    color = zero(Vec3{F})
    while depth > 0
        hit_result = hit(scene, ray)
        if hit_result.t == F(Inf)
            return color + accumulated_attenuation * background
        end
        
        ray, attenuation = scatter(ray, hit_result)
        color += accumulated_attenuation * emit(hit_result)
        if almost_zero(attenuation)
            return color
        end
        accumulated_attenuation *= attenuation
        depth -= 1
    end
    return color
end

function cu_path_trace(ray::Ray{F}, scene::CuSceneManager, buffer::CuDeviceArray; background::Vec3{F}=zero(Vec3{F}), depth::Int=50) where F<:AbstractFloat
    accumulated_attenuation = Vec3{F}(1)
    color = zero(Vec3{F})
    while depth > 0
        hit_result = cu_hit(buffer, scene, ray)
        if hit_result.t == Inf
            return color + accumulated_attenuation * background
        end
        
        ray, attenuation = scatter(ray, hit_result)
        color += accumulated_attenuation * emit(hit_result)
        if almost_zero(attenuation)
            return color
        end
        accumulated_attenuation *= attenuation
        depth -= 1
    end
    return color
end

function draw!(img::Array{RGB{F},2}, world::SceneManager, camera::Camera{F}; background::Vec3{F}=zero(Vec3{F}), pattern::Array{F, 2}=F.(MSAA_S16)) where F<:AbstractFloat
    height, width = size(img)
    
    bvh = build_bvh(world)
    
    for i = 0:width-1
        for j = height-1:-1:0
            col = Vec3{F}(0, 0, 0)
            for n in 1:size(pattern, 2)
                dx = pattern[1, n]
                dy = pattern[2, n]
                s = (i + dx) / (width-1)
                t = (j + dy) / (height-1)
                r = get_ray(camera, s, t)
                col += path_trace(r, bvh; background=background)
            end
            col /= size(pattern, 2)
            col = Vec3{F}(sqrt(col.x), sqrt(col.y), sqrt(col.z))
            img[height-j, i+1] = RGB{F}(col.x, col.y, col.z)
        end
        
        print("\r$i/$width")
    end
    
    print("\n")
    
    return img
end

function draw_kernel(img::CuDeviceArray{F}, scene::CuSceneManager, camera::Camera{F}, pattern::CuDeviceMatrix{F}, background::Vec3{F}, counter::CuDeviceArray, buffer::CuDeviceArray) where F
    CUDA.Const(pattern)
    
    stride = CUDA.CuStaticSharedArray(UInt32, 3)
    dims = CUDA.CuStaticSharedArray(UInt32, 3)
    
    y0 = UInt32((blockIdx()[1] - 1) * blockDim()[1] + threadIdx()[1])-1
    x0 = UInt32((blockIdx()[2] - 1) * blockDim()[2] + threadIdx()[2])-1
    z0 = UInt32((blockIdx()[3] - 1) * blockDim()[3] + threadIdx()[3])
    
    stride[1] = gridDim()[1]*blockDim()[1]
    stride[2] = gridDim()[2]*blockDim()[2]
    stride[3] = gridDim()[3]*blockDim()[3]
    
    dims[1] = size(img, 3)-1
    dims[2] = size(img, 2)-1
    dims[3] = size(pattern, 2)
    counter[2] = size(img, 2) * size(img, 3)
    
    for x = x0:stride[1]:dims[1], y = y0:stride[2]:dims[2]
        color = zero(Vec3{F})
        for z = z0:stride[3]:dims[3]
            s = (x + pattern[1, z]) / dims[1]
            t = (y + pattern[2, z]) / dims[2]
            
            r = get_ray(camera, s, t)
            color += cu_path_trace(r, scene, buffer; background=background)
        end
        CUDA.atomic_add!(pointer(img, (x * (dims[2]+1) + dims[2] - y) * 3 + 1), color.x)
        CUDA.atomic_add!(pointer(img, (x * (dims[2]+1) + dims[2] - y) * 3 + 2), color.y)
        CUDA.atomic_add!(pointer(img, (x * (dims[2]+1) + dims[2] - y) * 3 + 3), color.z)
        if z0 == 1
            CUDA.atomic_add!(pointer(counter, 1), 0x00000001)
            @cuprint "\r$(counter[1]/counter[2]*100)%"
        end
    end
    return nothing
end

function draw!(img::CuArray{RGB{F},2}, world::SceneManager, camera::Camera{F}; background::Vec3{F}=zero(Vec3{F}), pattern::Array{F, 2}=F.(MSAA_S16)) where F<:AbstractFloat
    height, width = size(img)
    
    bvh = build_bvh(world)
    cu_bvh = CuSceneManager(bvh)
    
    threads_dim = (16, 16, 1)
    blocks_dim = (8, 8, 1)
    buffer = CUDA.zeros(UInt16, 16, prod(threads_dim .* blocks_dim))
    
    CUDA.@time @cuda threads=threads_dim blocks=blocks_dim draw_kernel(channelview(img), cu_bvh, camera, CuArray(pattern), background, CUDA.zeros(UInt32, 2), buffer)
    img ./= size(pattern, 2)
    img .= map(c -> RGB{F}(sqrt(c.r), sqrt(c.g), sqrt(c.b)), img)
    synchronize()
    
    return img
end

function draw(world::Array{<:Hittable,1}, camera::Camera{F}, width::Int, height::Int; pattern::Array{F, 2}=F.(MSAA_S16)) where F<:AbstractFloat
    img = zeros(RGB{F}, height, width)
    draw!(world, camera, img, pattern)
end