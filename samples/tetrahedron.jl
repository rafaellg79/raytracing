scenes[:tetrahedron] = (F::Type=Float32; N::Int=500) -> begin
    aspect_ratio = F(3/2)
    width = 1200
    height = trunc(Int, width / aspect_ratio)
    
    # Camera
    origin = Vec3{F}(13, 2, 3)
    target = Vec3{F}(0, 0, 0)
    direction = origin - target
    up = Vec3{F}(0, 1, 0)
    vfov = F(20)
    hfov = rad2deg(atan(tan(deg2rad(vfov)/2)*aspect_ratio)*2)
    fov = hfov
    aperture = F(0.1)
    dist_to_focus = F(10)
    camera = Camera(origin, direction, Vec3{F}(0, 1, 0), F(fov), F(aspect_ratio), F(aperture), F(dist_to_focus))
    
    # World
    ground = Material(Lambertian, Vec3{F}(0.5, 0.5, 0.5))
    world = Hittable[Sphere(Vec3{F}(0, -1000, -1), 1000, ground)]
    for a = -11:10, b = -11:10
        choose_mat = rand(F)
        corner = Vec3{F}(2a + 0.9*rand(), 0.2, 2b + 0.9*rand())
        
        s = 0.25
        tetrahedron_vertices = [-1 -1 -1; -1 1 1; 1 -1 1; 1 1 -1]' .* s
        
        if length(corner - Vec3{F}(4, 0.2, 0)) > F(0.9)
            material = nothing
            if choose_mat < 0.8
                albedo = rand(Vec3{F}) * rand(Vec3{F})
                material = Material(Lambertian, albedo)
            elseif choose_mat < 0.95
                albedo = rand(Vec3{F})*F(0.5) + F(0.5)
                fuzz = rand(F)*F(0.5)
                material = Material(Metal, albedo, fuzz)
            else
                material = Material(Dielectric, F(1.5))
            end
            
            vertices = map(v -> Vec3{F}(v...) + corner, eachcol(RotXYZ(0, rand(F)*2pi, 0) * tetrahedron_vertices))
            
            # The ordering of the vertices below is important to determine the normal
            # which is important to compute hit information
            push!(world, Triangle(vertices[1], vertices[3], vertices[2], material))
            push!(world, Triangle(vertices[1], vertices[2], vertices[4], material))
            push!(world, Triangle(vertices[1], vertices[4], vertices[3], material))
            push!(world, Triangle(vertices[2], vertices[3], vertices[4], material))
        end
    end
    
    push!(world, Sphere(Vec3{F}(0, 1, 0), 1, Material(Dielectric, F(1.5))))
    push!(world, Sphere(Vec3{F}(-4, 1, 0), 1, Material(Lambertian, Vec3{F}(0.4, 0.2, 0.1))))
    push!(world, Sphere(Vec3{F}(4, 1, 0), 1, Material(Metal, Vec3{F}(0.7, 0.6, 0.5), zero(F))))
    
    background = Vec3{F}(0.70, 0.80, 1.00)
    pattern = Halton_sequence32(F, N)
    
    width, height, camera, world, background, pattern
end