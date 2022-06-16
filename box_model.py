import taichi as ti
import numpy as np

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda
ti.init(arch)
# ti.init(arch=ti.gpu,device_memory_GB=4)
# ti.init(ti.gpu)

dim = 3
n_grid = 100
steps = 5
dt = 2e-4

n_particles = 100000
print(n_particles)

dx = 1 / n_grid
inv_dx = n_grid

p_rho = 0.9
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
g_x = 0
g_y = -0
g_z = 0
bound = 3
E = 800  # Young's modulus
nu = 0.3  # Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(dim, ti.f32, n_particles)
fixed = ti.field(ti.i16, n_particles)
v = ti.Vector.field(dim, float, n_particles)
C = ti.Matrix.field(dim, dim, float, n_particles)
F = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # deformation gradient
Jp = ti.field(float, n_particles)

colors = ti.Vector.field(4, float, n_particles)
colors_random = ti.Vector.field(4, float, n_particles)
materials = ti.field(int, n_particles)
grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
grid_m = ti.field(float, (n_grid,) * dim)
grid_color = ti.Vector.field(4, float, (n_grid,) * dim)
grid_closest_particle = ti.field(ti.i32, shape=(n_grid,) * dim)
grid_closest_distance = ti.field(ti.f32, shape=(n_grid,) * dim)
used = ti.field(int, n_particles)

kernel_size = 3
neighbour = (kernel_size,) * dim

WATER = 0
JELLY = 1
SNOW = 2
BLOOD = 3

n_vertices_col = 18
n_verticles_row = 3
n_vertices = n_verticles_row * n_vertices_col
n_triangles = 2 * n_vertices_col * (n_verticles_row - 1)
vertices = ti.Vector.field(dim, ti.f32, n_vertices)
indices = ti.field(ti.i32, 3 * n_triangles)
color = (0.0, 0.0, 2.0)
per_vertice_color = ti.Vector.field(4, ti.f32, n_vertices)
rigid_particles = ti.Vector.field(dim, ti.f32, n_triangles)
center = ti.Vector.field(dim, ti.f32, shape=1)
PI = 3.1415926
mytool_radius = 0.08
kh = 0.5
r_v = 1.5
rigid_move = ti.Vector.field(dim, ti.f32, shape=())

grid_d = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))
grid_A = ti.field(dtype=int, shape=(n_grid, n_grid, n_grid))
grid_T = ti.field(dtype=int, shape=(n_grid, n_grid, n_grid))
grid_surface = ti.field(dtype=int, shape=(n_grid, n_grid, n_grid))

p_d = ti.field(dtype=float, shape=n_particles)
p_A = ti.field(dtype=int, shape=n_particles)
p_T = ti.field(dtype=int, shape=n_particles)
p_n = ti.Vector.field(2, dtype=float, shape=n_particles)

n_texture = 1024
texture = ti.Vector.field(4, ti.f32, n_texture ** 2)
# texture_xmin = 1000
# texture_xmax = -1
# texture_zmin = 1000
# texture_zmax = -1
# texture_ymin = 1000
# texture_ymax = -1

texture_xmin = 0.2
texture_xmax = 0.5
texture_zmin = 0.2
texture_zmax = 0.5
texture_ymin = 0.2
texture_ymax = 0.5

cubetool = ti.Vector.field(dim, ti.f32, shape=12)
cubetool_f = ti.field(ti.i32, 3 * 12)

diff_n_grid = 100
diff_dx = 1 / diff_n_grid
diff_inv_dx = float(diff_n_grid)
signed_distance_field = ti.field(ti.f32, shape=(diff_n_grid, diff_n_grid, diff_n_grid))
some_ids = ti.field(ti.int32, shape=(diff_n_grid, diff_n_grid, diff_n_grid)) # I don't know what these ids mean
face_num = ti.field(int, shape=())
et = ti.Vector.field(3, ti.i32, shape=256 * 4)
table_path ="./MC_Table.txt"
# table_path = "../datas/MC_Table.txt"
# table_path = "F:/GAMES101/aNewProject/marching_cube/MC_Table.txt"
mc_vertices = ti.Vector.field(dim, ti.f32, shape=15 * diff_n_grid ** 3)
per_mc_vertices_color = ti.Vector.field(4, ti.f32, shape=15 * diff_n_grid ** 3)
mc_triangles = ti.field(ti.i32, shape=15 * diff_n_grid ** 3)
diff_node_pos = ti.Vector.field(dim, ti.f32, shape=diff_n_grid ** 3)
per_diff_node_color = ti.Vector.field(dim, ti.f32, shape=diff_n_grid ** 3)


# @ti.kernel
# def init_particles():
#     for p in x:
#         x[p] = [ti.random() * 0.15 + 0.35, ti.random() * 0.15 + 0.25, ti.random() * 0.18 + 0.36]
#         colors[p] = ti.Vector([ti.random() * 0.5 + 0.5, 0.0, 0.0, 1.0])
#         # pass

def read_MCTable():
    f = open(table_path)
    count = 0
    for line in f:
        temp = line.split()
        for v in range(4):
            temp_index = v * 3
            x1 = int(temp[temp_index])
            x2 = int(temp[temp_index + 1])
            x3 = int(temp[temp_index + 2])
            et[count * 4 + v] = [x1, x2, x3]
        count += 1
    f.close()
    print(count)


@ti.kernel
def compute_implicit_face():
    for i, j, k in ti.ndrange(diff_n_grid, diff_n_grid, diff_n_grid):
        min_dis = 10.0
        node_pos = ti.Vector([i * diff_dx, j * diff_dx, k * diff_dx])
        for p in range(n_particles):
            distance = (x[p] - node_pos).norm() - 0.01
            if distance < min_dis:
                min_dis = distance
        signed_distance_field[i, j, k] = min_dis


@ti.func
def compute_face_vertice(i, j, k, edge):
    a = abs(signed_distance_field[i, j, k])
    b = abs(signed_distance_field[i + 1, j, k])
    c = abs(signed_distance_field[i + 1, j, k + 1])
    d = abs(signed_distance_field[i, j, k + 1])
    e = abs(signed_distance_field[i, j + 1, k])
    f = abs(signed_distance_field[i + 1, j + 1, k])
    g = abs(signed_distance_field[i + 1, j + 1, k + 1])
    h = abs(signed_distance_field[i, j + 1, k + 1])
    ac = grid_color[i, j, k]
    bc = grid_color[i + 1, j, k]
    cc = grid_color[i + 1, j, k + 1]
    dc = grid_color[i, j, k + 1]
    ec = grid_color[i, j + 1, k]
    fc = grid_color[i + 1, j + 1, k]
    gc = grid_color[i + 1, j + 1, k + 1]
    hc = grid_color[i, j + 1, k + 1]
    base_grid_pos = diff_dx * ti.Vector([i, j, k])
    result_pos = ti.Vector([0.0, 0.0, 0.0])
    result_color = ti.Vector([0.0, 0.0, 0.0, 0.0])
    if edge == 0:
        temp = a / (a + b)
        result_pos = base_grid_pos + ti.Vector([diff_dx * temp, 0, 0])
        if ac[3] > 0 and bc[3] > 0:
            result_color = temp * (ac - bc) + bc
        elif ac[3] > 0:
            result_color = ac
        else:
            result_color = bc
    if edge == 1:
        temp = b / (b + c)
        result_pos = base_grid_pos + ti.Vector([diff_dx, 0, diff_dx * temp])
        if bc[3] > 0 and cc[3] > 0:
            result_color = temp * (bc - cc) + cc
        elif bc[3] > 0:
            result_color = bc
        else:
            result_color = cc
    if edge == 2:
        temp = d / (c + d)
        result_pos = base_grid_pos + ti.Vector([diff_dx * temp, 0, diff_dx])
        if cc[3] > 0 and dc[3] > 0:
            result_color = temp * (cc - dc) + dc
        elif cc[3] > 0:
            result_color = cc
        else:
            result_color = dc
    if edge == 3:
        temp = a / (a + d)
        result_pos = base_grid_pos + ti.Vector([0, 0, diff_dx * temp])
        if ac[3] > 0 and dc[3] > 0:
            result_color = temp * (ac - dc) + dc
        elif ac[3] > 0:
            result_color = ac
        else:
            result_color = dc
    if edge == 4:
        temp = e / (e + f)
        result_pos = base_grid_pos + ti.Vector([diff_dx * temp, diff_dx, 0])
        if ec[3] > 0 and fc[3] > 0:
            result_color = temp * (ec - fc) + fc
        elif ec[3] > 0:
            result_color = ec
        else:
            result_color = fc
    if edge == 5:
        temp = f / (f + g)
        result_pos = base_grid_pos + ti.Vector([diff_dx, diff_dx, diff_dx * temp])
        if fc[3] > 0 and gc[3] > 0:
            result_color = temp * (fc - gc) + gc
        elif fc[3] > 0:
            result_color = fc
        else:
            result_color = gc
    if edge == 6:
        temp = h / (h + g)
        result_pos = base_grid_pos + ti.Vector([diff_dx * temp, diff_dx, diff_dx])
        if hc[3] > 0 and gc[3] > 0:
            result_color = temp * (hc - gc) + gc
        elif hc[3] > 0:
            result_color = hc
        else:
            result_color = gc
    if edge == 7:
        temp = e / (e + h)
        result_pos = base_grid_pos + ti.Vector([0, diff_dx, diff_dx * temp])
        if ec[3] > 0 and hc[3] > 0:
            result_color = temp * (ec - hc) + hc
        elif ec[3] > 0:
            result_color = ec
        else:
            result_color = hc
    if edge == 8:
        temp = a / (a + e)
        result_pos = base_grid_pos + ti.Vector([0, diff_dx * temp, 0])
        if ac[3] > 0 and ec[3] > 0:
            result_color = temp * (ac - ec) + ec
        elif ac[3] > 0:
            result_color = ac
        else:
            result_color = ec
    if edge == 9:
        temp = b / (b + f)
        result_pos = base_grid_pos + ti.Vector([diff_dx, diff_dx * temp, 0])
        if bc[3] > 0 and fc[3] > 0:
            result_color = temp * (bc - fc) + fc
        elif bc[3] > 0:
            result_color = bc
        else:
            result_color = fc
    if edge == 10:
        temp = c / (c + g)
        result_pos = base_grid_pos + ti.Vector([diff_dx, diff_dx * temp, diff_dx])
        if cc[3] > 0 and gc[3] > 0:
            result_color = temp * (cc - gc) + gc
        elif cc[3] > 0:
            result_color = cc
        else:
            result_color = gc
    if edge == 11:
        temp = d / (d + h)
        result_pos = base_grid_pos + ti.Vector([0, diff_dx * temp, diff_dx])
        if dc[3] > 0 and hc[3] > 0:
            result_color = temp * (dc - hc) + hc
        elif dc[3] > 0:
            result_color = dc
        else:
            result_color = hc
            # result_color = ti.Vector([0.8,0.0,0.0,1.0])
    return result_pos, result_color

@ti.kernel
def cal_ids_for_implict2explicit():
    for i, j, k in ti.ndrange(diff_n_grid - 1, diff_n_grid - 1, diff_n_grid - 1):
        id = 0
        if signed_distance_field[i, j, k] > 0:
            id |= 1
        if signed_distance_field[i + 1, j, k] > 0:
            id |= 2
        if signed_distance_field[i + 1, j, k + 1] > 0:
            id |= 4
        if signed_distance_field[i, j, k + 1] > 0:
            id |= 8
        if signed_distance_field[i, j + 1, k] > 0:
            id |= 16
        if signed_distance_field[i + 1, j + 1, k] > 0:
            id |= 32
        if signed_distance_field[i + 1, j + 1, k + 1] > 0:
            id |= 64
        if signed_distance_field[i, j + 1, k + 1] > 0:
            id |= 128
        some_ids[i, j, k] = id

@ti.kernel
def implicit_to_explicit():
    for i, j, k in ti.ndrange(diff_n_grid - 1, diff_n_grid - 1, diff_n_grid - 1):
        for ii in ti.static(range(4)):
            temp = some_ids[i, j, k] * 4 + ii
            if et[temp][0] != -1:
                # ti.ti_print(et[temp][0],et[temp][1],et[temp][2])
                n = ti.atomic_add(face_num[None], 1)
                f1, color1 = compute_face_vertice(i, j, k, et[temp][0])
                f2, color2 = compute_face_vertice(i, j, k, et[temp][1])
                f3, color3 = compute_face_vertice(i, j, k, et[temp][2])
                temp_index = n * 3
                mc_vertices[temp_index] = f1
                mc_vertices[temp_index + 1] = f2
                mc_vertices[temp_index + 2] = f3

                per_mc_vertices_color[temp_index] = color1
                per_mc_vertices_color[temp_index + 1] = color2
                per_mc_vertices_color[temp_index + 2] = color3

                mc_triangles[temp_index] = temp_index
                mc_triangles[temp_index + 1] = temp_index + 1
                mc_triangles[temp_index + 2] = temp_index + 2


###
def init_cubetool():
    center[0] = ti.Vector([0.52, 0.68, 0.67])
    phi = 8.0 / 14.0 * PI
    temp_x = center[0][0] + mytool_radius
    # temp_z = center[0][1]
    temp = ti.Vector([temp_x, center[0][1], center[0][2]])
    cubetool[0] = temp - ti.Vector([0.0, 0.0, 0.007])
    cubetool[1] = temp - ti.Vector([0.0, 0.007, 0.0])
    cubetool[2] = temp + ti.Vector([0.0, 0.0, 0.007])
    cubetool[3] = temp + ti.Vector([0.0, 0.007, 0.0])
    cubetool[4] = cubetool[0] + ti.Vector([0.03, 0.005, 0.0])
    cubetool[5] = cubetool[1] + ti.Vector([0.031, 0.005, 0.0])
    cubetool[6] = cubetool[2] + ti.Vector([0.03, 0.005, 0.0])
    cubetool[7] = cubetool[3] + ti.Vector([0.029, 0.005, 0.0])
    cubetool[8] = cubetool[0] + ti.Vector([0.3, 0.45, -0.001])
    cubetool[9] = cubetool[1] + ti.Vector([0.3, 0.45, 0.0])
    cubetool[10] = cubetool[2] + ti.Vector([0.3, 0.45, 0.001])
    cubetool[11] = cubetool[3] + ti.Vector([0.298, 0.45, 0.0])
    temp_index = 0
    for j in range(2):
        temp = j * 4
        for i in range(3):
            cubetool_f[temp_index] = temp + i
            temp_index += 1
            cubetool_f[temp_index] = temp + i + 1
            temp_index += 1
            cubetool_f[temp_index] = temp + i + 4
            temp_index += 1
            cubetool_f[temp_index] = temp + i + 1
            temp_index += 1
            cubetool_f[temp_index] = temp + i + 5
            temp_index += 1
            cubetool_f[temp_index] = temp + i + 4
            temp_index += 1
        cubetool_f[18] = temp + 3
        cubetool_f[19] = temp
        cubetool_f[20] = temp + 7
        cubetool_f[21] = temp
        cubetool_f[22] = temp + 4
        cubetool_f[23] = temp + 7


# def init_chuitiliu():
#     global texture_xmin, texture_xmax, texture_zmax, texture_zmin, texture_ymin, texture_ymax
#     chuitiliu_file = "../datas/500000.ply"
#     f = open(chuitiliu_file)
#     index_v = 0
#     for line in f:
#         temp = line.split()
#         temp_x = float(temp[0]) / 30.0 + 0.45
#         temp_y = float(temp[1]) / 30.0 - 4.65
#         temp_z = float(temp[2]) / 30.0
#         x[index_v] = [temp_x, temp_y, temp_z]
#         if temp_x < texture_xmin:
#             texture_xmin = temp_x
#         if temp_x > texture_xmax:
#             texture_xmax = temp_x
#         if temp_z < texture_zmin:
#             texture_zmin = temp_z
#         if temp_z > texture_zmax:
#             texture_zmax = temp_z
#         if temp_y < texture_ymin:
#             texture_ymin = temp_y
#         if temp_y > texture_ymax:
#             texture_ymax = temp_y
#         index_v += 1
#     f.close()


# def init_texture():
#     global texture_xmin, texture_xmax, texture_zmax, texture_zmin, texture_ymin, texture_ymax
#     texture_file = "../datas/rgb.txt"
#     f = open(texture_file)
#     index_t = 0
#     for line in f:
#         temp = line.split()
#         r = float(temp[0])
#         g = float(temp[1])
#         b = float(temp[2])
#         texture[index_t] = [r, g, b, 1.0]
#         index_t += 1


# @ti.kernel
# def init_uv():
#     temp_xsize = texture_xmax - texture_xmin
#     temp_ysize = texture_ymax - texture_ymin
#     temp_zsize = texture_zmax - texture_zmin
#     for p in x:
#         u = int((x[p][0] - texture_xmin) / temp_xsize * n_texture)
#         v = int((x[p][1] - texture_ymin) / temp_ysize * n_texture)
#         w = int((x[p][2] - texture_zmin) / temp_zsize * n_texture)
#         colors[p] = texture[u * n_texture + v]
#         # if x[p][1] > 0.5 and x[p][2] < 0.49:
#         #     colors[p] = ti.Vector([0.0,1.0,0.0,1.0])
@ti.kernel
def init_box():
    for p in x:
        # x[p] = [ti.random() - 0.5, ti.random() - 0.5,ti.random() - 0.5]
        x[p] = [ti.random()*0.2 + 0.3, ti.random()*0.2 + 0.3,ti.random()*0.2 + 0.3]

def init_texture():
    texture_file =  "./box_zhao_rgb.txt"
    f = open(texture_file)

    # For efficiency, first save all data into CPU memory
    cpu_texture = np.empty((texture.shape[0], 4), np.float32)
    index_t = 0
    lines = f.readlines()
    for line in lines:
        temp = line.split()
        r = float(temp[0])
        g = float(temp[1])
        b = float(temp[2])
        cpu_texture[index_t] = [r, g, b, 1.0]
        index_t += 1
    # Don't forget to close the file
    f.close()
    
    # Then copy from CPU to GPU
    texture.from_numpy(cpu_texture)


@ti.kernel
def init_uv():
    temp_xsize = texture_xmax - texture_xmin
    temp_ysize = texture_ymax - texture_ymin
    temp_zsize = texture_zmax - texture_zmin
    for p in x:
        u = int((x[p][0] - texture_xmin) / temp_xsize * n_texture)
        v = int((x[p][1] - texture_ymin) / temp_ysize * n_texture)
        w = int((x[p][2] - texture_zmin) / temp_zsize * n_texture)
        colors[p] = texture[v * n_texture + u]
        # colors[p] = ti.Vector([1.0,0.0,0.0,1.0])

@ti.kernel
def init_mytool():
    center[0] = ti.Vector([0.48, 0.68, 0.48])
    phi = 8.0 / 14.0 * PI
    for j in range(n_vertices_col):
        theta = 2.0 * float(j) / n_vertices_col * PI
        x = mytool_radius * ti.sin(phi) * ti.cos(theta)
        z = mytool_radius * ti.sin(phi) * ti.sin(theta)
        y = mytool_radius * ti.cos(phi)
        vertices[j] = ti.Vector([x, y, z]) + center[0]
        per_vertice_color[j] = ti.Vector([1.0, 1.0, 0.0, 1.0])
        # ti.ti_print(vertices[j])

    phi = 1.0 / 2.0 * PI
    for j in range(n_vertices_col):
        theta = 2.0 * float(j) / n_vertices_col * PI
        x = mytool_radius * ti.sin(phi) * ti.cos(theta)
        z = mytool_radius * ti.sin(phi) * ti.sin(theta)
        y = mytool_radius * ti.cos(phi)
        vertices[j + n_vertices_col] = ti.Vector([x, y, z]) + center[0]
        per_vertice_color[j + n_vertices_col] = ti.Vector([1.0, 1.0, 0.0, 1.0])
        # ti.ti_print(vertices[j])

    phi = 6.0 / 14.0 * PI
    for j in range(n_vertices_col):
        theta = 2.0 * float(j) / n_vertices_col * PI
        x = mytool_radius * ti.sin(phi) * ti.cos(theta)
        z = mytool_radius * ti.sin(phi) * ti.sin(theta)
        y = mytool_radius * ti.cos(phi)
        vertices[j + 2 * n_vertices_col] = ti.Vector([x, y, z]) + center[0]
        per_vertice_color[j + 2 * n_vertices_col] = ti.Vector([1.0, 1.0, 0.0, 1.0])
        # ti.ti_print(vertices[j])

    per_vertice_color[n_vertices_col - 1] = ti.Vector([1.0, 1.0, 1.0, 1.0])
    per_vertice_color[0] = ti.Vector([1.0, 0.0, 0.0, 1.0])
    per_vertice_color[2 * n_vertices_col - 1] = ti.Vector([0.0, 0.0, 1.0, 1.0])


def init_triangle():
    n_f = 0
    for i in range(n_verticles_row - 1):
        temp = i * n_vertices_col
        for j in range(n_vertices_col - 1):
            indices[n_f] = temp + j
            n_f += 1
            indices[n_f] = temp + j + n_vertices_col
            n_f += 1
            indices[n_f] = temp + j + 1
            n_f += 1
            indices[n_f] = temp + j + 1
            n_f += 1
            indices[n_f] = temp + j + n_vertices_col
            n_f += 1
            indices[n_f] = temp + j + n_vertices_col + 1
            n_f += 1

        indices[n_f] = temp + n_vertices_col - 1
        n_f += 1
        indices[n_f] = temp + n_vertices_col + n_vertices_col - 1
        n_f += 1
        indices[n_f] = temp
        n_f += 1
        indices[n_f] = temp
        n_f += 1
        indices[n_f] = temp + 2 * n_vertices_col - 1
        n_f += 1
        indices[n_f] = temp + n_vertices_col
        n_f += 1


@ti.kernel
def init_rigid_particles():
    for i in range(n_triangles):
        k = i * 3
        average = (vertices[indices[k]] + vertices[indices[k + 1]] + vertices[indices[k + 2]]) / 3
        rigid_particles[i] = average

        # for p in colors:
        #     colors[p] = ti.Vector([1.0,1.0,1.0,1.0])


@ti.func
def compute_plane_normal(surface):
    index = 3 * surface
    idx_a = indices[index]
    idx_b = indices[index + 1]
    idx_c = indices[index + 2]
    ab = vertices[idx_b] - vertices[idx_a]
    bc = vertices[idx_c] - vertices[idx_b]
    plane_normal = (ab.cross(bc)).normalized()
    return plane_normal


### 计算投影点
@ti.func
def compute_proj_point(surface, plane_normal, point):
    index = indices[3 * surface]
    plane_point = vertices[index]
    A = plane_normal[0]
    B = plane_normal[1]
    C = plane_normal[2]
    D = - A * plane_point[0] - B * plane_point[1] - C * plane_point[2]
    temp = float(A ** 2 + B ** 2 + C ** 2)
    proj_x = float((B ** 2 + C ** 2) * point[0] - A * (B * point[1] + C * point[2] + D)) / temp
    proj_y = float((A ** 2 + C ** 2) * point[1] - B * (A * point[0] + C * point[2] + D)) / temp
    proj_z = float((A ** 2 + B ** 2) * point[2] - C * (A * point[0] + B * point[1] + D)) / temp
    proj_point = ti.Vector([proj_x, proj_y, proj_z])
    return proj_point


### 计算距离
@ti.func
def compute_distance(point, proj_point):
    temp = point - proj_point
    return abs(temp.norm())


### 判断有效性
@ti.func
def is_valid(surface, proj_point):
    index = 3 * surface
    idx_a = indices[index]
    idx_b = indices[index + 1]
    idx_c = indices[index + 2]
    a = vertices[idx_a]
    b = vertices[idx_b]
    c = vertices[idx_c]
    ab = b - a
    bc = c - b
    ca = a - c
    ap = proj_point - a
    bp = proj_point - b
    cp = proj_point - c
    temp1 = ab.cross(ap)
    temp2 = bc.cross(bp)
    temp3 = ca.cross(cp)
    return temp1.dot(temp2) > 0 and temp2.dot(temp3) > 0


### 计算哪一侧
@ti.func
def compute_grid_T(plane_normal, point, proj_point, surface):
    temp = point - proj_point
    return plane_normal.dot(temp) > 0


@ti.func
def compute_particle_distance(particle_point):
    cp = particle_point - center[0]
    distance = cp.norm() - mytool_radius
    return distance


@ti.func
def compute_particle_norm(particle_point):
    pass


### 圆环下移
@ti.kernel
def substep_rigidMove_updateVertices():
    for p in vertices:
        vertices[p] += dt * r_v * rigid_move[None]
@ti.kernel
def substep_rigidMove_updateParticles():
    for p in rigid_particles:
        rigid_particles[p] += dt * r_v * rigid_move[None]
### Grid CDF
@ti.kernel
def substep_CDF_initGrid():
    for i, j, k in grid_A:
        grid_A[i, j, k] = 0
        grid_T[i, j, k] = 0
        grid_d[i, j, k] = 0.0
        grid_surface[i, j, k] = -1
        grid_color[i, j, k] = ti.Vector([0.0, 0.0, 0.0, 0.0])
        grid_closest_distance[i, j, k] = 10
@ti.kernel
def substep_CDF_updateGrid():
    for p in rigid_particles:
        base = (rigid_particles[p] * inv_dx - 0.5).cast(int)
        for i, j, k in ti.static(ti.ndrange(kernel_size, kernel_size, kernel_size)):
            offset = ti.Vector([i, j, k])
            grid_node = (offset + base).cast(float) * dx
            plane_normal = compute_plane_normal(p)
            proj_point = compute_proj_point(p, plane_normal, grid_node)
            if is_valid(p, proj_point):
                grid_A[base + offset] = 1
                distance = compute_distance(grid_node, proj_point)
                if grid_surface[base + offset] == -1 or grid_d[base + offset] > distance:
                    grid_d[base + offset] = distance
                    grid_surface[base + offset] = p
                    if compute_grid_T(plane_normal, grid_node, proj_point, p) == True:
                        grid_T[base + offset] = 1
                    else:
                        grid_T[base + offset] = -1
### Particle CDF
@ti.kernel
def substep_CDF_updateParticle():
    for p in x:
        p_A[p] = 0
        p_T[p] = 0
        p_d[p] = 0.0

        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        Tpr = 0.0
        for i, j, k in ti.static(
                ti.ndrange(kernel_size, kernel_size, kernel_size)):  # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j, k])
            if grid_A[base + offset] == 1:
                p_A[p] = 1
            weight = w[i][0] * w[j][1] * w[k][2]
            Tpr += weight * grid_d[base + offset] * grid_T[base + offset]
        p_d[p] = abs(Tpr)
        if p_A[p] == 1:
            p_d[p] = compute_particle_distance(x[p])
            if Tpr > 0:
                p_T[p] = 1
                # colors[p] = ti.Vector([1.0,0.0,0.0,1.0])
            else:
                p_T[p] = -1
                # colors[p] = ti.Vector([0.0,1.0,0.0,1.0])
@ti.kernel
def substep_initGrid_vm():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
@ti.kernel
def substep_P2G():
    '''maybe P2G, not sure'''
    for p in x:
        if used[p] == 0:
            continue
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]  # deformation gradient update
        h = ti.exp(10 * (1.0 - Jp[p]))  # Hardening coefficient: snow gets harder when compressed
        if materials[p] == JELLY:  # jelly, make it softer
            # h = 0.08
            pass
        mu, la = mu_0 * 0.3, lambda_0 * 0.3
        # if materials[p] == BLOOD:
        #     mu *= 0.6

        if materials[p] == WATER:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if materials[p] == SNOW:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if materials[p] == WATER:  # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F[p] = new_F
        elif materials[p] == SNOW:
            F[p] = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (
            J - 1)
        stress = (-dt * p_vol * 4) * stress / dx ** 2
        affine = stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            if p_T[p] * grid_T[base + offset] == -1:
                pass
            else:
                dpos = (offset.cast(float) - fx) * dx
                weight = 1.0
                for i in ti.static(range(dim)):
                    weight *= w[offset[i]][i]

                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
                grid_color[base + offset] += weight * p_mass * colors[p]
            # gpdis = (x[p] - (base + offset) * dx).norm()
            # if gpdis < grid_closest_distance[base + offset]:
            #     grid_closest_distance[base + offset] = gpdis
            #     grid_color[base + offset] = colors[p]
                # grid_closest_particle = p
                # grid_color[base + offset] += colors[p]
                # temp11 = base + offset
                # ti.ti_print(temp11)
                # ti.ti_print(weight * colors[p])
@ti.kernel
def substep_updateGrid_v():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] /= grid_m[i, j, k]
            grid_color[i,j,k] /= grid_m[i, j, k]
        grid_v[i, j, k] += dt * ti.Vector([g_x, g_y, g_z])
        if i < bound and grid_v[i, j, k][0] < 0:
            grid_v[i, j, k][0] = 0
        if i > n_grid - bound and grid_v[i, j, k][0] > 0:
            grid_v[i, j, k][0] = 0
        if j < bound - 2 and grid_v[i, j, k][1] < 0:
            grid_v[i, j, k][1] = 0
        if j > n_grid - bound and grid_v[i, j, k][1] > 0:
            grid_v[i, j, k][1] = 0
        if k < bound and grid_v[i, j, k][2] < 0:
            grid_v[i, j, k][2] = 0
        if k > n_grid - 1 and grid_v[i, j, k][2] > 0:
            grid_v[i, j, k][2] = 0
        # if j > fixed_y and k < fixed_z:
        #     grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def substep_G2P():
    '''maybe G2P, not sure'''
    for p in x:
        if used[p] == 0:
            continue
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(v[p])
        new_C = ti.zero(C[p])

        cp = x[p] - center[0]
        np = cp.normalized() * p_T[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            g_v = ti.Vector([0.0, 0.0, 0.0])
            if p_T[p] * grid_T[base + offset] == -1:
                sg = v[p].dot(np)
                if sg > 0:
                    g_v = v[p]
                else:
                    g_v = v[p] - v[p].dot(np) * np
                if p_T[p] * p_d[p] > 0:
                    # g_v += np * (5 * dx - abs(p_d[p])) * 15
                    g_v += np * 2
                materials[p] = BLOOD
                # colors[p] = ti.Vector([0.0,0.0,1.0,1.0])
            else:
                g_v = grid_v[base + offset]

            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx ** 2

        v[p] = new_v
        if p_A[p] and p_T[p] * p_d[p] < 0:
            f_penalty = - kh * np * p_d[p] * 5 * p_T[p]
            v[p] += dt * f_penalty / p_mass

        # if fixed[p] == 0:
        x[p] += dt * v[p]
        C[p] = new_C

def substep(g_x: float, g_y: float, g_z: float):
    substep_rigidMove_updateVertices()
    substep_rigidMove_updateParticles()
    center[0] += dt * r_v * rigid_move[None]

    substep_CDF_initGrid()
    substep_CDF_updateGrid()

    substep_CDF_updateParticle()

    substep_initGrid_vm()

    substep_P2G()

    fixed_y = (0.5 * inv_dx)
    fixed_z = (0.49 * inv_dx)
    # for p in x:
    #     if x[p][1] > 0.27 and x[p][2] < 0.57:
    #         colors[p] = ti.Vector([1.0,0.0,0.0,1.0])

    substep_updateGrid_v()

    # ti.block_dim(n_grid)
    substep_G2P()




class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


@ti.kernel
def init_cube_vol(first_par: int, last_par: int, x_begin: float,
                  y_begin: float, z_begin: float, x_size: float, y_size: float,
                  z_size: float, material: int):
    for i in range(first_par, last_par):
        # x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector(
        #     [x_size, y_size, z_size]) + ti.Vector([x_begin, y_begin, z_begin])
        Jp[i] = 1
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        materials[i] = material
        colors_random[i] = ti.Vector(
            [ti.random(), ti.random(),
             ti.random(), ti.random()])
        used[i] = 1


@ti.kernel
def set_all_unused():
    for p in used:
        used[p] = 0
        # basically throw them away so they aren't rendered
        x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        Jp[p] = 1
        F[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        v[p] = ti.Vector([0.0, 0.0, 0.0])


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume
    next_p = 0
    for i in range(len(vols)):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(vols) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size, v.material)
            next_p += par_count
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(material_colors: ti.ext_arr()):
    for i in range(n_particles):
        mat = materials[i]
        colors[i] = ti.Vector([
            material_colors[mat, 0], material_colors[mat, 1],
            material_colors[mat, 2], 1.0
        ])


@ti.kernel
def print_grid_color():
    for i, j, k in ti.ndrange(diff_n_grid, diff_n_grid, diff_n_grid):
        ti.ti_print(grid_color[i, j, k])


print("Loading presets...this might take a minute")

presets = [[CubeVolume(ti.Vector([0.2, 0.0, 0.65]),
                       ti.Vector([0.3, 0.3, 0.3]), JELLY), ]]
preset_names = [
    "Single Dam Break",
    "Double Dam Break",
    "Water Snow Jelly",
]

curr_preset_id = 0
paused = False
use_random_colors = False
particles_radius = 0.005
material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]


def init():
    global paused
    init_vols(presets[curr_preset_id])


init()

res = (1920, 1080)
# window = ti.ui.Window("Real MPM 3D", res, vsync=True,show_window=False)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)
frame_id = 0
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.6)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)


def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))

    colors_used = colors_random if use_random_colors else colors
    # scene.particles(x, per_vertex_color=colors, radius=particles_radius)
    # scene.particles(x, per_vertex_color=colors, radius=particles_radius)

    scene.mesh(vertices=vertices, indices=indices, color=(0.75, 0.75, 0.75, 1.0), two_sided=True)
    # scene.mesh(cubetool, cubetool_f, color=(0.75, 0.75, 0.75, 1.0), two_sided=True)

    scene.mesh(vertices=mc_vertices, indices=mc_triangles, per_vertex_color=per_mc_vertices_color)

    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.2, 0.2, 0.2))
    scene.point_light(pos=(0.5, 1.0, 1.8), color=(0.2, 0.2, 0.2))
    canvas.scene(scene)


from MySimpleTimer import MySimpleTimer
timer = MySimpleTimer()

timer.tick()
init_mytool()
print("init_mytool():", str(timer.tick()))
init_triangle()
print("init_triangle():", str(timer.tick()))
init_rigid_particles()
print("init_rigid_particles():", str(timer.tick()))

print("cube...")
init_texture()
print("init_texture():", str(timer.tick()))
init_box()
print("init_box():", str(timer.tick()))

print("chuitiliu")
init_uv()
print("init_uv():", str(timer.tick()))
read_MCTable()
print("read_MCTable():", str(timer.tick()))
rigid_move[None] = ti.Vector([0.0, -2.0, 0.0])
print("rigid_move[None] = ti.Vector([0.0, -2.0, 0.0]):", str(timer.tick()))

print("start running...")
while window.running:
    compute_implicit_face()
    print("compute_implicit_face():", str(timer.tick()))
    cal_ids_for_implict2explicit()
    print("cal_ids_for_implict2explicit()", str(timer.tick()))
    implicit_to_explicit()
    print("implicit_to_explicit():", str(timer.tick()))
    if not paused:
        for s in range(steps):
            substep(g_x, g_y, g_z)
            print("substep(g_x, g_y, g_z):", str(timer.tick()))
    render()
    print("render():", str(timer.tick()))
    window.show()
    print("window.show():", str(timer.tick()))
    if frame_id == 300:
        rigid_move[None] = ti.Vector([0.0, 0.0, 0.0])
    if frame_id == 550 :
        break
    frame_id += 1
    face_num[None] = 0
