import Euler1D
import numpy as np
import matplotlib.pyplot as plt
import RiemannExact as RE

# Specify the control file locations
path_minmod = "./HLLCTest_minmod/test"
path_VanLeer = "./HLLCTest_VanLeer/test"
path_zero = "./HLLCTest_zero/test"

path_list = [path_minmod, path_VanLeer, path_zero]
name_list = ["minmod", "Van Leer", "0-order"]
save_loc = "./ImgMUSCL/test"
y_list = [r"$\rho$", r"$u$", r"$p$"]

# Define the problems to solve (For the exact Godunov solver)
problems_dic = {
    "stateL": [
        [1.0, 0.0, 1.0],
        [1.0, -2.0 ,0.4],
        [1.0, 0.0, 1000.0],
        [1.0, 0.0, 0.01],
        [5.99924, 19.5975, 460.894],
    ],
    "stateR": [
        [0.125, 0.0, 0.1],
        [1.0, 2.0, 0.4],
        [1.0, 0.0, 0.01],
        [1.0, 0.0, 100.0],
        [5.99242, -6.19633, 46.0950],
    ],
    "x0": [
        0.5, 0.5, 0.5, 0.5, 0.4
    ],
    "t0": [
        0.25, 0.15, 0.012, 0.035, 0.035
    ]
}

ne = 201
x = np.linspace(0, 1, ne)

for j in range(5): # go through all the problems
    print(f"*************** Solving Problem {j} ****************")
    file0 = path_list[0]+str(j+1)+".txt"
    file1 = path_list[1]+str(j+1)+".txt"
    file2 = path_list[2]+str(j+1)+".txt"

    solver0 = Euler1D.Euler1D(file0)
    solver1 = Euler1D.Euler1D(file1)
    solver2 = Euler1D.Euler1D(file2)

    rp = RE.RiemannExact(np.array(problems_dic["stateL"][j]), 
        np.array(problems_dic["stateR"][j]), gamma = 1.4
    )
    t0 = problems_dic["t0"][j]
    x0 = problems_dic["x0"][j]

    #Solve the approximate Riemann Problem
    cfl = 0.10
    global_t = 0.0
    while global_t < t0:
        dt = solver0.time_advancement_RK3(cfl = cfl)
        global_t += dt
    solver0.con2Prim()

    cfl = 0.10
    global_t = 0.0
    while global_t < t0:
        dt = solver1.time_advancement_RK3(cfl = cfl)
        global_t += dt
    solver1.con2Prim()

    cfl = 0.10
    global_t = 0.0
    while global_t < t0:
        dt = solver2.time_advancement_RK3(cfl = cfl)
        global_t += dt
    solver2.con2Prim()

    # Solve the exact Riemann problem
    rp.solve()
    s = (x - x0) / t0
    Esol = np.zeros((3, s.shape[0]))
    for col, ss in enumerate(s):
        Esol[:, col] = rp.sample(ss)
    # plot and save the result
    fig, ax = plt.subplots(1,3, figsize = (15,4), dpi = 200, sharex = True)
    y_list = [r"$\rho$", r"$u$", r"$p$", r"$e_{int}$"]
    for k in range(3):
        ax[k].plot(solver0.mesh, solver0.sol[k, :], label = "Minmod")
        ax[k].plot(solver1.mesh, solver1.sol[k, :], label = "Van Leer")
        ax[k].plot(solver2.mesh, solver2.sol[k, :], label = "0-order")
        ax[k].plot(x, Esol[k], label = "Exact", linestyle = "dashed")
        ax[k].legend()
        ax[k].grid()
        ax[k].set_xlabel("$x$")
        ax[k].set_ylabel(y_list[k])
        ax[k].set_xlim(0, 1)
    
    fig.savefig(save_loc+str(j+1)+".png")
    print(f"*************** Completed {j} ****************\n")