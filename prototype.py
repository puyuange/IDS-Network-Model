import pandas as pd
import matplotlib.pyplot as plt

# to store the data
rowlist = []

# Initial values
N_A = 10000
N_B = 10000
I_A = 100
I_B = 0
R_A = 0
R_B = 0
S_A = N_A - I_A - R_A
S_B = N_B - I_B - R_B
beta = 0.2
gamma = 0.1
k = 5
dayrange = 160


# Calculate new state every day (dt = 1)
def change(S_A, I_A, R_A, S_B, I_B, R_B, beta, gamma, k):
    N_A = S_A + I_A + R_A
    N_B = S_B + I_B + R_B
    j = k/N_A

    # Susceptible
    dS_A = -(beta * I_A * S_A)/N_A - j * S_A
    dS_B = -(beta * I_B * S_B)/N_B + j * S_A

    # Infected
    dI_A = (beta * I_A * S_A)/N_A - gamma * I_A - j * I_A
    dI_B = (beta * I_B * S_B)/N_B - gamma * I_B + j * I_A

    # Recovered
    dR_A = gamma * I_A - j * R_A
    dR_B = gamma * I_B + j * R_A

    # Change
    S_A = S_A + dS_A
    I_A = I_A + dI_A
    R_A = R_A + dR_A
    S_B = S_B + dS_B
    I_B = I_B + dI_B
    R_B = R_B + dR_B

    return [S_A, I_A, R_A, S_B, I_B, R_B]


for i in range(0, dayrange):
    # Record
    print(str(i) + ", " + str(S_A) + ", " + str(I_A) + ", " + str(R_A) + ", " + str(S_B) + ", " + str(I_B) + ", " + str(R_B))
    row = {"t": i, "S_A": S_A, "I_A": I_A, "R_A": R_A, "S_B": S_B, "I_B": I_B, "R_B": R_B}
    rowlist.append(row)

    # Change
    vals = change(S_A, I_A, R_A, S_B, I_B, R_B, beta, gamma, k)
    S_A = vals[0]
    I_A = vals[1]
    R_A = vals[2]
    S_B = vals[3]
    I_B = vals[4]
    R_B = vals[5]

df = pd.DataFrame(rowlist)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax2.set_xlabel("time (days)")
ax1.set_ylabel("number")
ax2.set_ylabel("number")

ax1.plot(df["t"], df["S_A"], color="blue", label="S_A")
ax1.plot(df["t"], df["I_A"], color="red", label="I_A")
ax1.plot(df["t"], df["R_A"], color="green", label="R_A")
ax1.legend(loc="upper right", ncol=3)

ax2.plot(df["t"], df["S_B"], color="blue", label="S_B")
ax2.plot(df["t"], df["I_B"], color="red", label="I_B")
ax2.plot(df["t"], df["R_B"], color="green", label="R_B")
ax2.legend(loc="upper right", ncol=3)

plt.show()

