import taichi as ti

# 初始化 Taichi
ti.init(arch=ti.gpu)

# 图像分辨率
width, height = 800, 600

# 像素颜色缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))

# UI 参数绑定的全局变量
ka = ti.field(dtype=ti.f32, shape=())
kd = ti.field(dtype=ti.f32, shape=())
ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())

# 设置默认值
ka[None] = 0.2
kd[None] = 0.7
ks[None] = 0.5
shininess[None] = 32.0

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """计算射线与球体的交点"""
    oc = ro - center
    a = rd.dot(rd)
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    discriminant = b * b - 4.0 * a * c
    
    t = 1e10
    N = ti.Vector([0.0, 0.0, 0.0])
    
    if discriminant >= 0.0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        if t1 > 0.0:
            t = t1
        elif t2 > 0.0:
            t = t2
        
        if t < 1e10:
            hit_point = ro + t * rd
            N = (hit_point - center).normalized()
    
    return t, N

@ti.func
def intersect_cone(ro, rd, vertex, base_y, base_radius):
    """计算射线与有限高度圆锥的交点"""
    t = 1e10
    N = ti.Vector([0.0, 0.0, 0.0])
    
    height = vertex.y - base_y
    k = base_radius / height
    k2 = k * k
    
    co = ro - vertex
    
    a = rd.x**2 + rd.z**2 - k2 * rd.y**2
    b = 2.0 * (co.x * rd.x + co.z * rd.z - k2 * co.y * rd.y)
    c = co.x**2 + co.z**2 - k2 * co.y**2
    
    discriminant = b * b - 4.0 * a * c
    
    if discriminant >= 0.0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        for t_cand in ti.static([t1, t2]):
            if 0.0 < t_cand < t:
                p = ro + t_cand * rd
                if base_y <= p.y <= vertex.y:
                    t = t_cand
                    cy = p.y - vertex.y
                    N = ti.Vector([p.x - vertex.x, -k2 * cy, p.z - vertex.z]).normalized()
                    
    return t, N

@ti.kernel
def render():
    for i, j in pixels:
        u = i / width
        v = j / height
        
        cam_pos = ti.Vector([0.0, 0.0, 5.0])
        aspect = width / height
        x = (2.0 * u - 1.0) * aspect
        y = 2.0 * v - 1.0
        
        rd = ti.Vector([x, y, -1.0]).normalized()
        ro = cam_pos
        
        sphere_center = ti.Vector([-1.2, -0.2, 0.0])
        sphere_radius = 1.2
        sphere_color = ti.Vector([0.8, 0.1, 0.1])
        
        cone_vertex = ti.Vector([1.2, 1.2, 0.0])
        cone_base_y = -1.4
        cone_base_radius = 1.2
        cone_color = ti.Vector([0.6, 0.2, 0.8])
        
        t_min = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_object = 0 
        
        t_sphere, N_sphere = intersect_sphere(ro, rd, sphere_center, sphere_radius)
        if 0.0 < t_sphere < t_min:
            t_min = t_sphere
            hit_normal = N_sphere
            hit_point = ro + t_sphere * rd
            hit_object = 1
        
        t_cone, N_cone = intersect_cone(ro, rd, cone_vertex, cone_base_y, cone_base_radius)
        if 0.0 < t_cone < t_min:
            t_min = t_cone
            hit_normal = N_cone
            hit_point = ro + t_cone * rd
            hit_object = 2
        
        if t_min < 1e10:
            light_pos = ti.Vector([2.0, 3.0, 4.0])
            
            N = hit_normal.normalized()
            L = (light_pos - hit_point).normalized()
            V = (cam_pos - hit_point).normalized()
            
            # 【进阶 1：Blinn-Phong 模型】 计算半程向量 H
            H = (L + V).normalized()
            
            Ka = ka[None]
            Kd = kd[None]
            Ks = ks[None]
            Shin = shininess[None]
            
            base_color = sphere_color if hit_object == 1 else cone_color
            
            # 环境光 (无论是否在阴影中都有)
            Ambient = Ka * base_color

            # 漫反射
            diffuse_intensity = ti.max(N.dot(L), 0.0)
            Diffuse = Kd * diffuse_intensity * base_color
                
            # 镜面反射
            specular_intensity = ti.max(N.dot(H), 0.0)
            Specular = Ks * ti.pow(specular_intensity, Shin) * base_color
            color = Ambient + Diffuse + Specular
            Specular = Ks * ti.pow(specular_intensity, Shin) * base_color
            specular_intensity = ti.max(N.dot(H), 0.0)
            Specular = Ks * ti.pow(specular_intensity, Shin) * ti.Vector([1.0, 1.0, 1.0])
            
            color = Ambient + Diffuse + Specular
            color = ti.math.clamp(color, 0.0, 1.0)
            pixels[i, j] = color
        else:
            color = ti.Vector([0.0, 0.2, 0.3])
            pixels[i, j] = color

def main():
    window = ti.ui.Window("Taichi Ray Casting Lab (Blinn-Phong & Hard Shadow)", (width, height))
    canvas = window.get_canvas()
    
    while window.running:
        render()
        
        window.GUI.begin("Material Parameters", 0.05, 0.05, 0.3, 0.25)
        ka[None] = window.GUI.slider_float("Ka (Ambient)", ka[None], 0.0, 1.0)
        kd[None] = window.GUI.slider_float("Kd (Diffuse)", kd[None], 0.0, 1.0)
        ks[None] = window.GUI.slider_float("Ks (Specular)", ks[None], 0.0, 1.0)
        shininess[None] = window.GUI.slider_float("Shininess", shininess[None], 1.0, 128.0)
        window.GUI.end()
        
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main()