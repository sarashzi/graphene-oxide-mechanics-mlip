from mace.calculators import mace_mp
from ase.md.npt import NPT
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write
from ase import units
from ase.md import MDLogger
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import Stationary
from ase.md.velocitydistribution import ZeroRotation

#----- parameters --------------------
strain_rate = 1e-6 #fs
dt_fs = 0.5
dt = dt_fs * units.fs 
system_name = 'GO-20-1'
n_steps_npt = 100000

T_init = 300
total_strain = 0.17 # shall be changed based on diffrent oxidation and OH/O ratio in GO

macemp = mace_mp(dispersion=True) # I use for oh and O effect 
atoms = read('/examples/go_20_oxidation_OH_O_1.data' , format= 'lammps-data') # GO with 20 % oxidation and OH/O = 1
atoms.calc = macemp

#---- minimazation ------------------
print("starting minimizing energy")
dyn = BFGS(atoms)
dyn.run(fmax = 0.05, steps = 100) 

write(f"{system_name}_min.lammps-data", atoms)
print("Optimization done!")

# initialize velocitie
MaxwellBoltzmannDistribution(atoms, temperature_K=T_init)
Stationary(atoms)
ZeroRotation(atoms)


#--------- NPT equilbirum ---------------------
mask = (0, 1, 0)

dyn = NPT(atoms,
          timestep = dt,
          temperature_K = T_init,
          externalstress= 0.0 ,
          mask = mask,
          ttime = 10 * units.fs,
          pfactor = 1000 * units.fs)



def write_frame_npt():
    atoms.write(
        f"dump_npt_{system_name}.xyz",
        append=True,
    )


print('Strating NPT-MD')
dyn.attach(write_frame_npt, interval = 100) # 
dyn.attach(ZeroRotation, interval=50, atoms=atoms)
dyn.attach(MDLogger(dyn, atoms, f"npt_{system_name}.log", stress = True, peratom =False, mode = "a"), 
           interval = 100)
dyn.run(n_steps_npt)
write(f"{system_name}_after_npt.lammps-data", atoms)
print("Equilbrium (NPT) done!") 

           
#----------NVT (Deformation step) -------------------
print("Starting NVT and uniaxial deformation in y direction")

cell0 = atoms.get_cell()
ly0 = cell0[1,1]
print(f"the ly0 is: {ly0}")


n_steps_def = int(total_strain / (strain_rate * dt_fs)) 
print(f"Deformation: target strain={total_strain}, "
     f"strain_rate={strain_rate}/fs, timestep ={dt_fs} fs so --> number of steps={n_steps_def}")

# ---- strain log file ----
strain_log = f"strain_{system_name}.dat"
with open(strain_log, "w") as f:
    f.write("# step   strain_y\n")

    
# deformation function
current_strain = 0.0
step_idx = 0  

def deform_step():
    global current_strain, step_idx
    current_strain += strain_rate * dt_fs
    if current_strain > total_strain:
        return  
    Ly_new = ly0 * (1.0 + current_strain)
    cell = atoms.get_cell()
    cell[1, 1] = Ly_new
    atoms.set_cell(cell, scale_atoms=True)
    
    with open(strain_log, "a") as f:
        f.write(f"{step_idx} {current_strain}\n")
    step_idx+=1    
    

def write_frame_nvt():
    atoms.write(
        f"dump_nvt_{system_name}.xyz",
        append=True,
    )


dyn_def = Langevin(atoms, dt, temperature_K=T_init, friction =0.001/units.fs)
dyn_def.attach(deform_step, interval=1)
dyn_def.attach(write_frame_nvt, interval=100)
dyn_def.attach(ZeroRotation, interval=50, atoms=atoms)
dyn_def.attach(MDLogger(dyn_def, atoms, f"nvt_{system_name}.log", stress = True, peratom =False, mode = "a"),
               interval = 100)          
dyn_def.run(n_steps_def)


print("Uniaxial deformation done!")
write(f"{system_name}_final_def.lammps-data", atoms)

