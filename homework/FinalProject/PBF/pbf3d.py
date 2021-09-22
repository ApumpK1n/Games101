# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.

import math

import numpy as np

import taichi as ti
import tina

ti.init(arch=ti.gpu)

screen_res = (800, 400, 400)
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio,
            screen_res[2] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1))

dim = 3
bg_color = 0xffffff
particle_color = 0x00BFFF
boundary_color = 0xebaca2
num_particles_x = 2
num_particles_y = 10
num_particles_z = 1
num_particles = num_particles_x * num_particles_y * num_particles_z
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 0
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
neighbor_radius = h * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

gravity = ti.Vector([0.0, -9.8, 0.0])

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
board_states_right = ti.Vector.field(dim, float)
board_states_left = ti.Vector.field(dim, float)


ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
grid_snode = ti.root.dense(ti.ijk, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.l, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.place(board_states_right)
ti.root.place(board_states_left)


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    bmin = ti.Vector([board_states_left[None][0] + particle_radius_in_world, 0, 0]) 
    bmax = ti.Vector([board_states_right[None][0], boundary[1], boundary[2]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin[i]:
            p[i] = bmin[i] + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def move_board():
    #使用边界模拟施加力的情况
    right = board_states_right[None]
    right[1] += 1.0
    period = 90
    vel_strength = 8.0
    if right[1] >= 2 * period:
        right[1] = 0
    right[0] += -ti.sin(right[1] * np.pi / period) * vel_strength * time_delta
    board_states_right[None] = right

    left = board_states_left[None]
    left[1] += 1.0
    if left[1] >= 2 * period:
        left[1] = 0
    left[0] += ti.sin(left[1] * np.pi / period) * vel_strength * time_delta
    board_states_left[None] = left


@ti.kernel
def prologue():
    # 存储旧位置
    for i in positions:
        old_positions[i] = positions[i]
    # 重力
    for i in positions:
        pos, vel = positions[i], velocities[i]
        vel += gravity * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)

    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        # Eq(11)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, h)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...


def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


def render(gui):
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.rect((0, 0), (board_states_right[None][0] / boundary[0], 1),
             radius=1.5,
             color=boundary_color)
    gui.rect((board_states_left[None][0] / boundary[0], 1), (0, 0),
            radius=1.5,
            color=boundary_color)
    gui.show()


@ti.kernel
def init_particles():
    print("init")
    for i in range(num_particles):
        delta = h * 0.5
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02, 0.0])
        positions[i] = ti.Vector([i % num_particles_x, i // (num_particles_x * num_particles_z), 
        (i - ((i // (num_particles_x * num_particles_z)) * (num_particles_x * num_particles_z))) // num_particles_x ]) * delta + offs
        
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states_right[None] = ti.Vector([boundary[0] - epsilon, -0.0, 0.0])
    board_states_left[None] = ti.Vector([0 + epsilon, -0.0, 0.0])


def print_stats():
    #print('PBF stats:')
    num = grid_num_particles.to_numpy()
    avg, max = np.mean(num), np.max(num)
    #print(f'  #particles per cell: avg={avg:.2f} max={max}')
    num = particle_num_neighbors.to_numpy()
    avg, max = np.mean(num), np.max(num)
    #print(f'  #neighbors per particle: avg={avg:.2f} max={max}')
    print(f'left:{board_states_left[None][0]}, right:{board_states_right[None][0]} boundary={boundary}' )

scene = tina.Scene((800, 400), maxpars=num_particles, bgcolor=ti.hex_to_rgb(0xaaaaff))
pars = tina.SimpleParticles(num_particles, radius=particle_radius)
color = tina.Diffuse(color=ti.hex_to_rgb(0xffaaaa))
scene.add_object(pars, color)

gui = ti.GUI('PBF3D', scene.res)
scene.init_control(gui, center=[0.5, 0.5, 0.5], radius=1.5)

scene.lighting.clear_lights()
scene.lighting.add_light([-0.4, 1.5, 1.8], color=[0.8, 0.8, 0.8])
scene.lighting.set_ambient_light([0.22, 0.22, 0.22])

def main():
    init_particles()
    #print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    #gui = ti.GUI('PBF2D', screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        
        move_board()
        run_pbf()
        if gui.frame % 20 == 1:
            print_stats()
        
        scene.input(gui)
        scene.render()
        pos = positions.to_numpy()
        pars.set_particles(pos)
        pars.set_particle_radii(np.ones(len(pos), dtype=np.float32) * particle_radius)
        gui.set_image(scene.img)
        gui.show()
    


if __name__ == '__main__':
    main()
