scenes[:cu_cover2] = (F::Type=Float32) -> begin
    aspect_ratio = one(F)
    width = 800
    height = trunc(Int, width / aspect_ratio)
    
    # Camera
    origin = Vec3{F}(478, 278, -600)
    target = Vec3{F}(278, 278, 0)
    direction = origin - target
    up = Vec3{F}(0, 1, 0)
    vfov = F(40)
    hfov = rad2deg(atan(tan(deg2rad(vfov)/2)*aspect_ratio)*2)
    fov = hfov
    aperture = zero(F)
    dist_to_focus = F(10)
    camera = Camera(origin, direction, Vec3{F}(0, 1, 0), fov, aspect_ratio, aperture, dist_to_focus, zero(F), one(F))
    
    # Materials
    ground = Material(Lambertian, Vec3{F}(0.48, 0.83, 0.53))
    moving_sphere_material = Material(Lambertian, Vec3{F}(0.7, 0.3, 0.1))
    white = Material(Lambertian, Vec3{F}(0.73))
    light = Material(Emissive, Vec3{F}(7, 7, 7))
    
    # For now we don't support perlin nor image textures, so we will
    # render a solid color sphere instead
    missing_texture = Material(Lambertian, Vec3{F}(0, 1, 1))
    
    # Boxes in ground
    boxes_per_side = 20
    
    boxes = Vector{Box{F, Material{F, SolidColor{F}}}}(undef, boxes_per_side*boxes_per_side)
    for i = 1:boxes_per_side, j = 1:boxes_per_side
        n = (i - 1) * boxes_per_side + j
        w = 100
        x0 = -1000 + i * w
        z0 = -1000 + j * w
        y0 = 0
        x1 = x0 + w
        y1 = rand(F) * 79 + 1 # Reduced the height to avoid a box covering a sphere depending on the rand seed
        z1 = z0 + w
        
        boxes[n] = Box(Vec3{F}(x0, y0, z0), Vec3{F}(x1, y1, z1), ground)
    end
    
    # Spheres in scene
    objects = Hittable[]
    
    # Moving sphere at top left
    center1 = Vec3{F}(400, 400, 200)
    center2 = center1 + Vec3{F}(30, 0, 0)
    push!(objects, MovingSphere(center1, center2, 0, 1, 50, moving_sphere_material))
    
    # Dielectric (glass-like) sphere at bottom
    push!(objects, Sphere(Vec3{F}(260, 150, 45), 50, Material(Dielectric, F(1.5))))
    
    # Metal sphere at far right
    push!(objects, Sphere(Vec3{F}(0, 150, 145), 50, Material(Metal, Vec3{F}(0.8, 0.8, 0.9), one(F))))
    
    # Blue sphere at the lower left
    boundary = Sphere(Vec3{F}(360, 150, 145), 70, Material(Dielectric, F(1.5)))
    push!(objects, boundary)
    push!(objects, ConstantMedium(boundary, F(0.2), Vec3{F}(0.2, 0.4, 0.9)))
    
    # Fog on the background
    boundary = Sphere(zero(Vec3{F}), 5000, Material(Dielectric, F(1.5)))
    push!(objects, ConstantMedium(boundary, F(0.0001), Vec3{F}(1)))
    
    # Earth
    push!(objects, Sphere(Vec3{F}(400, 200, 400), 100, missing_texture))
    
    # Perlin texture
    push!(objects, Sphere(Vec3{F}(220, 280, 300), 80, missing_texture))
    
    # Spheres in box at top right corner
    ns = 1000
    spheres = [Sphere(rand(Vec3{F}, zero(F), F(165.0)), 10, white) for _ = 1:ns]
    
    # World
    world = Hittable[
                     xz_rect(123, 423, 147, 412, 554, light);
                     objects;
                     boxes;
                     Transform.(spheres, Ref(F[RotY(deg2rad(15)) [-100; 270; 395]; 0 0 0 1]));
                    ]
    
    background = zero(Vec3{F})
    pattern = Halton_sequence32(F, 1000)
    
    width, height, camera, world, background, pattern
end