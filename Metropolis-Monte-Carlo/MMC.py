import matplotlib.pyplot as plt
import numpy as np

# general metropolis monte-carlo method
def metropolis(old_energy, new_energy,kt):
    if new_energy < old_energy:
        return new_energy # accept if new energy is lower
    
    p = np.exp(-(new_energy - old_energy) / kt)
    if np.random.rand() <= p:
        return new_energy # accept new state
    else:
        return old_energy # reject new state
    
    
# case of single particle with spin 0 or 1
def one_particle(n_steps,kt):
    energies = np.zeros(n_steps)
    running_avg = np.zeros(n_steps)
    
    for i in range(1,n_steps):
        new_energy = int(1-energies[i-1]) # attempt to flip state
        energies[i] = metropolis(energies[i-1],new_energy,kt)
        running_avg[i] = np.mean(energies[:i+1])
    
    return energies, running_avg


# run for two values of kt and save figures
for kt in [1,2]:
    energies, running_avg = one_particle(100,kt)
    
    plt.figure()
    plt.plot(energies,label='State')
    plt.plot(running_avg,label='Running average')
    plt.title(f'kt = {kt}')
    plt.ylabel('Energy')
    plt.xlabel('Steps')
    plt.legend()
    plt.savefig(f'one_particle_kt={kt}.png')