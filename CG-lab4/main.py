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
    """计算射线与有限高度圆锥的交点（修复并补全）"""
    t = 1e10
    N = ti.Vector([0.0, 0.0, 0.0])
    
    # 圆锥的高度 (顶点y - 底面y)
    height = vertex.y - base_y
    
    # k = 半径 / 高度
    k = base_radius / height
    k2 = k * k
    
    # 将射线转到以圆锥顶点为原点的坐标系下计算
    co = ro - vertex
    
    # 圆锥侧面的隐式方程: x^2 + z^2 = k^2 * y^2
    a = rd.x**2 + rd.z**2 - k2 * rd.y**2
    b = 2.0 * (co.x * rd.x + co.z * rd.z - k2 * co.y * rd.y)
    c = co.x**2 + co.z**2 - k2 * co.y**2
    
    discriminant = b * b - 4.0 * a * c
    
    if discriminant >= 0.0:
        sqrt_disc = ti.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        
        # 遍历两个交点，找到最近的且在高度范围内的合法交点
        for t_cand in ti.static([t1, t2]):
            if 0.0 < t_cand < t:
                p = ro + t_cand * rd
                # 检查交点是否在截断范围内：底面高度 <= p.y <= 顶点高度
                if base_y <= p.y <= vertex.y:
                    t = t_cand
                    # 利用梯度求圆锥侧面的法向量
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
        
        # --- 按照题目要求更新坐标和颜色 ---
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
        
        # 深度竞争
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
            R = (2.0 * N.dot(L) * N - L).normalized()
            
            # --- 修复：从全局 field 读取 UI 控制的材质参数 ---
            Ka = ka[None]
            Kd = kd[None]
            Ks = ks[None]
            Shin = shininess[None]
            
            base_color = sphere_color if hit_object == 1 else cone_color
            
            Ambient = Ka * base_color
            
            diffuse_intensity = ti.max(N.dot(L), 0.0)
            Diffuse = Kd * diffuse_intensity * base_color
            
            specular_intensity = ti.max(R.dot(V), 0.0)
            Specular = Ks * ti.pow(specular_intensity, Shin) * ti.Vector([1.0, 1.0, 1.0])
            
            color = Ambient + Diffuse + Specular
            color = ti.math.clamp(color, 0.0, 1.0)
            pixels[i, j] = color
        else:
            # 背景色：深青色
            color = ti.Vector([0.0, 0.2, 0.3])
            pixels[i, j] = color
        

# --- 修复：补充创建窗口与主循环 ---
def main():
    window = ti.ui.Window("Taichi Ray Casting Lab", (width, height))
    canvas = window.get_canvas()
    
    while window.running:
        # 1. 执行渲染 Kernel
        render()
        
        # 2. 绘制 UI 面板
        window.GUI.begin("Material Parameters", 0.05, 0.05, 0.3, 0.25)
        ka[None] = window.GUI.slider_float("Ka (Ambient)", ka[None], 0.0, 1.0)
        kd[None] = window.GUI.slider_float("Kd (Diffuse)", kd[None], 0.0, 1.0)
        ks[None] = window.GUI.slider_float("Ks (Specular)", ks[None], 0.0, 1.0)
        shininess[None] = window.GUI.slider_float("Shininess", shininess[None], 1.0, 128.0)
        window.GUI.end()
        
        # 3. 将像素输出到屏幕并显示
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main()