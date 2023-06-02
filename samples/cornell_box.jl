scenes[:cornell_box] = (F::Type=Float32) -> begin
    aspect_ratio = one(F)
    width = 400
    height = trunc(Int, width / aspect_ratio)
    
    # Camera
    origin = Vec3{F}(278, 278, -800)
    target = Vec3{F}(278, 278, 0)
    direction = origin - target
    up = Vec3{F}(0, 1, 0)
    vfov = F(40)
    hfov = rad2deg(atan(tan(deg2rad(vfov)/2)*aspect_ratio)*2)
    fov = hfov
    aperture = zero(F)
    dist_to_focus = F(10)
    camera = Camera(origin, direction, Vec3{F}(0, 1, 0), fov, aspect_ratio, aperture, dist_to_focus, zero(F), one(F))
    
    # World
    red   = Material(Lambertian, Vec3{F}(0.65, 0.05, 0.05))
    white = Material(Lambertian, Vec3{F}(0.73, 0.73, 0.73))
    green = Material(Lambertian, Vec3{F}(0.12, 0.45, 0.15))
    light = Material(Emissive, Vec3{F}(15, 15, 15))
    world = Hittable[
                    yz_rect(0, 555, 0, 555, 555, green),
                    yz_rect(0, 555, 0, 555, 0, red),
                    xz_rect(213, 343, 227, 332, 554, light),
                    xz_rect(0, 555, 0, 555, 0, white),
                    xz_rect(0, 555, 0, 555, 555, white),
                    xy_rect(0, 555, 0, 555, 555, white),
                    Transform(Box(zero(Vec3{F}), Vec3{F}(165, 330, 165), white), F[RotY(deg2rad(0)) [265; 0; 295]; 0 0 0 1]),
                    Transform(Box(zero(Vec3{F}), Vec3{F}(165, 165, 165), white), F[RotY(deg2rad(-18)) [130; 0; 65]; 0 0 0 1]),
                ]
    
    background = zero(Vec3{F})
    pattern = Halton_sequence32(F, 1000)
    
    width, height, camera, world, background, pattern
end