struct Vec3{T}
    x::T
    y::T
    z::T
    Vec3{T}(x::T) where T = new(x, x, x)
    Vec3{T}(x::F) where {F, T} = new(T(x), T(x), T(x))
    Vec3{T}(v::Vec3{F}) where {F, T} = new(T(v.x), T(v.y), T(v.z))
    Vec3{T}(x::Real, y::Real, z::Real) where T = new(T(x), T(y), T(z))
end

# Define some useful functions for the Vec3 type
import Base: +, -, *, /, ==, <, <=, >, >=, min, max, zero, rand, length

zero(::Type{Vec3{F}}) where F = Vec3{F}(zero(F), zero(F), zero(F))

# Vector arithmetic
+(a::Vec3{F}, b::Vec3{F}) where F = Vec3{F}(a.x + b.x, a.y + b.y, a.z + b.z)
-(a::Vec3{F}, b::Vec3{F}) where F = Vec3{F}(a.x - b.x, a.y - b.y, a.z - b.z)
*(a::Vec3{F}, b::Vec3{F}) where F = Vec3{F}(a.x * b.x, a.y * b.y, a.z * b.z)
/(a::Vec3{F}, b::Vec3{F}) where F = Vec3{F}(a.x / b.x, a.y / b.y, a.z / b.z)

# Unary operations
+(a::Vec3{F}) where F = a
-(a::Vec3{F}) where F = Vec3{F}(-a.x, -a.y, -a.z)

# Scalar arithmetic
+(a::Vec3{F}, b::Real) where F = Vec3{F}(a.x + b, a.y + b, a.z + b)
-(a::Vec3{F}, b::Real) where F = Vec3{F}(a.x - b, a.y - b, a.z - b)
*(a::Vec3{F}, b::Real) where F = Vec3{F}(a.x * b, a.y * b, a.z * b)
/(a::Vec3{F}, b::Real) where F = a * F(1 / b)

+(a::Real, b::Vec3{F}) where F = Vec3{F}(a + b.x, a + b.y, a + b.z)
-(a::Real, b::Vec3{F}) where F = Vec3{F}(a - b.x, a - b.y, a - b.z)
*(a::Real, b::Vec3{F}) where F = Vec3{F}(a * b.x, a * b.y, a * b.z)
/(a::Real, b::Vec3{F}) where F = Vec3{F}(a / b.x, a / b.y, a / b.z)

# Comparison operations
 <(a::Vec3{F}, b::Vec3{F}) where F = a.x <  b.x && a.y <  b.y && a.z <  b.z
<=(a::Vec3{F}, b::Vec3{F}) where F = a.x <= b.x && a.y <= b.y && a.z <= b.z
 >(a::Vec3{F}, b::Vec3{F}) where F = a.x >  b.x && a.y >  b.y && a.z >  b.z
>=(a::Vec3{F}, b::Vec3{F}) where F = a.x >= b.x && a.y >= b.y && a.z >= b.z
==(a::Vec3{F}, b::Vec3{F}) where F = a.x == b.x && a.y == b.y && a.z == b.z

# Dot product
dot(a::Vec3{F}, b::Vec3{F}) where F = a.x * b.x + a.y * b.y + a.z * b.z

# Cross product
cross(a::Vec3{F}, b::Vec3{F}) where F = Vec3{F}(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x)

# Vector length
function length(a::Vec3{F}) where F
    return sqrt(dot(a, a))
end

# Normalize vector
function normalize(a::Vec3{F}) where F
    return a / length(a)
end

# Zero comparison
function almost_zero(a::Vec3{F}) where F
    return isapprox(a.x, 0) && isapprox(a.y, 0) && isapprox(a.z, 0)
end

# Min and max of two Vectors
function min(a::Vec3{F}, b::Vec3{F}) where F
    return Vec3{F}(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z))
end

function max(a::Vec3{F}, b::Vec3{F}) where F
    return Vec3{F}(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z))
end

function minmax(a::Vec3{F}, b::Vec3{F}) where F
    x0, x1 = minmax(a.x , b.x)
    y0, y1 = minmax(a.y , b.y)
    z0, z1 = minmax(a.z , b.z)
    return Vec3{F}(x0, y0, z0), Vec3{F}(x1, y1, z1)
end

# Random
rand(::Type{Vec3{F}}) where F = Vec3{F}(rand(F), rand(F), rand(F))
rand(::Type{Vec3{F}}, s::Int) where F = [rand(Vec3{F}) for i in 1:s]
rand(::Type{Vec3{F}}, min::F, max::F) where F = Vec3{F}(rand(F) * (max - min) + min, rand(F) * (max - min) + min, rand(F) * (max - min) + min)

function random_in_unit_sphere(::Type{Vec3{F}}=Vector3f) where F
    ONE = one(F)
    p = rand(Vec3{F}, -ONE, ONE)
    while dot(p, p) > ONE
        p = rand(Vec3{F}, -ONE, ONE)
    end
    return p
end

function random_in_unit_disk(::Type{Vec3{F}}=Vector3f) where F
    ONE = one(F)
    ZERO = zero(F)
    TWO = F(2)
    p = Vec3{F}(rand(F) * TWO - ONE, rand(F) * TWO - ONE, ZERO)
    while dot(p, p) > ONE
        p = Vec3{F}(rand(F) * TWO - ONE, rand(F) * TWO - ONE, ZERO)
    end
    return p
end

function random_unit_vector(::Type{V}=Vector3f) where V<:Vec3
    return normalize(random_in_unit_sphere(V))
end