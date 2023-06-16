using Rotations

scenes[:cornell_smoke_box] = (;F::Type=Float32, spp::Int=200) -> begin
    aspect_ratio = one(F)
    width = 600
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
    light = Material(Emissive, Vec3{F}(7, 7, 7))
    
    box1 = Box(zero(Vec3{F}), Vec3{F}(165, 330, 165), white)
    box1 = Transform(box1, F[RotY(deg2rad(15)) [265; 0; 295]; 0 0 0 1])
    
    box2 = Box(zero(Vec3{F}), Vec3{F}(165, 165, 165), white)
    box2 = Transform(box2, F[RotY(deg2rad(-18)) [130; 0; 65]; 0 0 0 1])
    
    world = Hittable[
                    yz_rect(0, 555, 0, 555, 555, green),
                    yz_rect(0, 555, 0, 555, 0, red),
                    xz_rect(113, 443, 127, 432, 554, light),
                    xz_rect(0, 555, 0, 555, 0, white),
                    xz_rect(0, 555, 0, 555, 555, white),
                    xy_rect(0, 555, 0, 555, 555, white),
                    ConstantMedium(box1, 0.01, Vec3{F}(0.0)),
                    ConstantMedium(box2, 0.01, Vec3{F}(1.0)),
                ]
    
    background = zero(Vec3{F})
    pattern = Halton_sequence32(F, spp)
    
    width, height, camera, world, background, pattern
end