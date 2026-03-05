import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.optimize import curve_fit
from scipy.stats import t


""" error in measurements """
# Volume
V_meas = np.array([])

# Volume uncertainty
V_err = np.array([])

# Measured time
t_meas = np.array([])

t_err = 0.2  # time error 

# Pressure drop
deltaP = np.array([])

# Pressure uncertainty
P_err = np.array([])  
# 4.9 Pa for analogue, 1000 Pa for digital


mu = 1 #Use value from Omni Calc
rho = 997 #Use value from Omni Calc
D = 0.017
area = np.pi * D**2 / 4




""" flow rate/ uncertainty"""
Q = V_meas / t_meas

Q_err = Q * ((V_err / V_meas) +(t_err / t_meas))

"""Velocity/uncertainty"""

velocity = Q / area
velocity_err = velocity * (Q_err / Q)

epsilon = 0.0001   

Re = (rho * velocity * D) / mu
f_exp = (2 * deltaP * D) / (rho * velocity**2)



"""reynolds error"""
Re_err = Re * (velocity_err / velocity)

"""friction factor error"""
f_err = f_exp *((P_err / deltaP) + (2 * velocity_err / velocity))


""" Only keep turbulent data to avoid fitting the laminar point"""


turbulent = Re > 2000
Re_turb = Re[turbulent]
f_turb = f_exp[turbulent]


"""Haaland equation"""

def haaland(Re, rel_rough):
    return 1 / (-1.8*np.log10((rel_rough/3.7)**1.11 + 6.9/Re))**2


""" Curve fitting the line to the points we have"""

def model(Re, rel_rough):
    return haaland(Re, rel_rough)

initial_guess = [0.001]

params, covariance = curve_fit(model, Re_turb, f_turb, p0=initial_guess)

rel_rough_fit = params[0]
std_error = np.sqrt(np.diag(covariance))[0]


""" Calculating the r^2 value of the fitted curve"""

f_pred = haaland(Re_turb, rel_rough_fit)

ss_res = np.sum((f_turb - f_pred)**2)
ss_tot = np.sum((f_turb - np.mean(f_turb))**2)

r_squared = 1 - ss_res / ss_tot

""" Confidence interval of values"""
n = len(f_turb)
dof = n - 1

t_val = t.ppf(0.975, dof)

ci_low = rel_rough_fit - t_val * std_error
ci_high = rel_rough_fit + t_val * std_error


"""Moody curves """
Re_laminar = np.logspace(2, np.log10(2000), 200)
f_laminar = 64 / Re_laminar

Re_range = np.logspace(np.log10(2000), 7, 400)

f_smooth = haaland(Re_range, 0)
f_rough = haaland(Re_range, epsilon/D)
f_fit_curve = haaland(Re_range, rel_rough_fit)


"""Plot the moody curves, laminar region, transition region etc"""


plt.figure(figsize=(10, 7))

plt.loglog(Re_laminar, f_laminar, label="Laminar flow", linewidth=3)
plt.loglog(Re_range, f_smooth, label="Smooth pipe")
plt.loglog(Re_range, f_rough, label="ε = 0.1 mm", linewidth=3)

plt.loglog(Re_range, f_fit_curve, 'r--', linewidth=3,
           label="Fitted roughness curve")




plt.errorbar(Re, f_exp, 
    xerr=Re_err,
    yerr=f_err,
    fmt='o',
    color='red',
    ecolor='black',
    capsize=3,
    label="Experimental data")


plt.axvspan(2000, 4000, color="grey", alpha=0.2)

plt.xlabel("Reynolds Number")
plt.ylabel("Darcy Friction Factor")
plt.title("Moody Diagram with Fitted Roughness Curve")

plt.grid(True, which="both", linestyle="--")
plt.legend()

plt.xlim(1e2, 1e7)
plt.ylim(0.008, 0.2)

plt.savefig(r'FILELOCATION', dpi=300)

plt.show()


"""Print r^2, CI, Rel roughness, Absolute roughness"""

print(f"Relative roughness (ε/D) = {rel_rough_fit:.6f}")

print(f"95% CI for ε/D = [{ci_low:.6f}, {ci_high:.6f}]")

print(f"Absolute roughness ε = {rel_rough_fit * D:.6e} m")

print(f"R² = {r_squared:.5f}")

""" add data to CSV"""

data = { 
    'Volume': V_meas,
    'Volume error': V_err,
    'Time': t_meas,
    'Pressure difference': deltaP,
    'Pressure error': P_err,
    'Friction factor': f_exp,
    'Reynolds number': Re,
    'Temperature': 26}

df =  pd.DataFrame(data)

df.to_csv(r'FILELOCATION')
