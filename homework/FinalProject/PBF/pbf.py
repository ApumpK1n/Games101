import math
import numpy as np
import taichi as ti


ti.init(arch=ti.gpu)

#粒子
num_particles_x = 60
num_particles_y = 20
#粒子总数
num_particles = num_particles_x * num_particles_y
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio
cell_size = 2.51
cell_recpr = 1.0 / cell_size


#场景
#维度
dimension = 2
time_delta = 1.0 / 20.0
screen_size = (800, 400)
screen_to_world_ratio = 10.0
boundary = (screen_size[0] / screen_to_world_ratio, screen_size[1] / screen_to_world_ratio)
epsilon = 0

#taichi params
old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
board_states = ti.Vector.field(2, float)
grid_num_particles = ti.field(int)
particle_neighbors = ti.field(int)
grid2particles = ti.field(int)


h = 1.1
mass = 1.0
neighbor_radius = h * 1.05

@ti.func
def poly6_value(r, h):
    result = 0.0
    if 0 < r and r < h:
        x = (h * h - r * r) / (h * h * h)
        result = (315.0 / 64.0 / math.pi) * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor =  (-45.0 / math.pi) * x * x
        result = r * g_factor / r_len
    return result


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x
                                  ]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(cell):
    return 0 <= cell[0] and cell[0] < grid_size[0] and 0 <= cell[1] and cell[
        1] < grid_size[1]


@ti.kernel
def preDealData():
    # 存储位置信息
    for i in positions:
        old_positions[i] = positions[i]
    
    # 增加重力
    for i in positions:
        g = ti.Vector([0.0, -9.8])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

def pbf():
    preDealData()

def main():
    init_particles()
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    gui = ti.GUI('pbf', screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        move_board()
        pbf()
        if gui.frame % 20 == 1:
            print_stats()
        render(gui)


if __name__ == '__main__':
    main()
