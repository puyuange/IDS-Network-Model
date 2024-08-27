import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# SEIR model equations
def SEIR_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Initial conditions
S0 = 0.99  
E0 = 0.01  
I0 = 0.00  
R0 = 0.00  
y0 = [S0, E0, I0, R0]

# Parameters
beta = 0.3    
sigma = 0.1   
gamma = 0.05 

# Time vector
t = np.linspace(0, 500, 500)  

# Solve the SEIR model equations
solution = odeint(SEIR_model, y0, t, args=(beta, sigma, gamma))

# Extract results
S, E, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.title('SEIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()