# Type II controller for two-input system

import numpy as np
import matplotlib.pyplot as plt
import math as math

import pyit2fls
from pyit2fls import T1FS, T1FS_plot, T1Mamdani
from pyit2fls import IT2FS, IT2FLS, IT2FS_plot, trapezoid_mf, tri_mf, TR_plot, crisp, min_t_norm, max_s_norm, product_t_norm

domain = np.linspace(-2, 2, 1001)
domain2 = np.linspace(0, 1, 251)
domain3 = np.linspace(-1, 1, 501)

class5 = IT2FLS()

mid5 = IT2FS(domain, tri_mf, [-0.3, 0, 0.3, 1], tri_mf, [-0.1, 0, 0.1, 0.7])
litneg5 = IT2FS(domain, trapezoid_mf, [-1.7, -1.5, -0.2, 0, 1], trapezoid_mf, [-1.5, -1.3, -0.4, -0.2, 0.7])
litpos5 = IT2FS(domain, trapezoid_mf, [0, 0.2, 1.5, 1.7, 1], trapezoid_mf, [0.2, 0.4, 1.3, 1.5, 0.7])
bigpos5 = IT2FS(domain, trapezoid_mf, [1.1, 1.6, 2.1, 3.1, 1], trapezoid_mf, [1.3, 1.8, 2.3, 3.3, 0.7])
bigneg5 = IT2FS(domain, trapezoid_mf, [-3.1, -2.1, -1.6, -1.1, 1], trapezoid_mf, [-3.3, -2.3, -1.8, -1.3, 0.7])
IT2FS_plot(bigneg5, litneg5, mid5, litpos5, bigpos5, title="Input1: Error value", legends=["big negative", "little negative", "negligible", "little positive", "big positive"], xlabel="Domain", ylabel="Value")
class5.add_input_variable("x51")

neg5 = IT2FS(domain3, trapezoid_mf, [-1.2, -1, -0.3, -0.005, 1], trapezoid_mf, [-1.1, -1, -0.5, -0.015, 0.7])
nul5 = IT2FS(domain3, tri_mf, [-0.025, 0, 0.025, 1], tri_mf, [-0.015, 0, 0.015, 0.7])
poz5 = IT2FS(domain3, trapezoid_mf, [0.005, 0.3, 1, 1.2, 1], trapezoid_mf, [0.015, 0.5, 1, 1.1, 0.7])

IT2FS_plot(neg5, nul5, poz5, title="Input2: Output derivation value", legends=["negative", "zero", "positive"], xlabel="Domain", ylabel="Value")
class5.add_input_variable("x52")

middle5 = IT2FS(domain2, tri_mf, [0.4, 0.5, 0.6, 1], tri_mf, [0.45, 0.5, 0.55, 0.7])
parcls5 = IT2FS(domain2, trapezoid_mf, [0.05, 0.15, 0.35, 0.45, 1], trapezoid_mf, [0.125, 0.225, 0.275, 0.375,  0.7])
paropn5 = IT2FS(domain2, trapezoid_mf, [0.55, 0.65, 0.85, 0.95, 1], trapezoid_mf, [0.625, 0.725, 0.775 , 0.875, 0.7])
fulopn5 = IT2FS(domain2, trapezoid_mf, [0.75, 0.9, 1.1, 1.2, 1], trapezoid_mf, [0.825, 0.975, 1.1, 1.3, 0.7])
fulcls5 = IT2FS(domain2, trapezoid_mf, [-0.6, -0.2, 0.1, 0.25, 1], trapezoid_mf, [-0.4, -0.1, 0.025, 0.175, 0.7])
IT2FS_plot(fulcls5, parcls5, middle5, paropn5, fulopn5, title="Output: Vent pipe position", legends=["fully closed", "partially opened", "middle opened", "very opened" ,"fully opened"], xlabel="Domain", ylabel="Value")
class5.add_output_variable("y51")


class5.add_rule([("x51", mid5), ("x52", neg5)], [("y51", middle5)])
class5.add_rule([("x51", mid5), ("x52", nul5)], [("y51", middle5)])
class5.add_rule([("x51", mid5), ("x52", poz5)], [("y51", middle5)])
class5.add_rule([("x51", litneg5), ("x52", neg5)], [("y51", parcls5)])
class5.add_rule([("x51", litneg5), ("x52", nul5)], [("y51", parcls5)])
class5.add_rule([("x51", litneg5), ("x52", poz5)], [("y51", middle5)])
class5.add_rule([("x51", litpos5), ("x52", neg5)], [("y51", middle5)])
class5.add_rule([("x51", litpos5), ("x52", nul5)], [("y51", paropn5)])
class5.add_rule([("x51", litpos5), ("x52", poz5)], [("y51", paropn5)])
class5.add_rule([("x51", bigneg5), ("x52", neg5)], [("y51", fulcls5)])
class5.add_rule([("x51", bigneg5), ("x52", nul5)], [("y51", fulcls5)])
class5.add_rule([("x51", bigneg5), ("x52", poz5)], [("y51", parcls5)])
class5.add_rule([("x51", bigpos5), ("x52", neg5)], [("y51", paropn5)])
class5.add_rule([("x51", bigpos5), ("x52", nul5)], [("y51", fulopn5)])
class5.add_rule([("x51", bigpos5), ("x52", poz5)], [("y51", fulopn5)])

domainn=np.linspace(-2, 2, 101)
domainn2=np.linspace(-1, 1, 101)

Z=np.ones((domainn.shape[0], domainn2.shape[0]))
for i, x1 in zip(range(len(domainn)),domainn):
	for j, x2 in zip(range(len(domainn2)),domainn2):
		it2out, tr=class5.evaluate({"x51":x1, "x52":x2}, min_t_norm, max_s_norm, domain)
		Z[i][j]=crisp(tr["y51"])

X1, X2 = np.meshgrid(domen, domen2)

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X1, X2, Z)
ax.set_xlabel("Error")
ax.set_ylabel("Derivative")
ax.set_zlabel("Vent pipe")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("3D rule base surface")
plt.show()

g=9.81
lb=0
ub=2*1
area=1
S=0.05
lower=0.5*np.ones(200)
upper=1.5*np.ones(200)
input=np.hstack((upper, lower, upper, lower, upper, lower, upper, lower, upper, lower))
error5=np.zeros(2000)
output5=np.zeros(2001)
outputs5=np.zeros(2000)
derivation=np.zeros(2001)
output5[0]=0.5
derivation[0]=0
flow_out=0.1565
first=0.5
i=-1
previous=0.5
for t in range(1,2000,1):
  t=t/20
  i=i+1
  error5[i]=input[i]-output5[i]
  it2out, tr=class5.evaluate({"x51":error5[i], "x52":derivation[i]}, min_t_norm, max_s_norm, domain, method="Centroid")
  fuzzy_output=crisp(tr["y51"])
  outputs5[i]=fuzzy_output
  if fuzzy_output<0:
    fuzzy_output=0
  elif fuzzy_output>1:
    fuzzy_output=1
  scaled=fuzzy_output*0.5
  new=scaled-flow_out
  if (new and ((first>lb) or (new>=0)) and ((first<ub) or (new<=0))):
    temp=new
  else:
    temp=0
  first=temp*0.1+previous
  previous=first
  if (first<lb):
    limited=lb
  elif (first>ub):
    limited=ub
  else:
    limited=first
  output5[i+1]=limited/area
  derivation[i+1]=(output5[i+1]-output5[i])/0.1
  if (derivation[i+1]<-1):
    derivation[i+1]=-1
  elif (derivation[i+1]>1):
    derivation[i+1]=1
  velocity=math.sqrt(2*g*output5[i+1])
  flow_out=velocity*S

t=np.linspace(0,200,2001)

plt.plot(t[0:2000],input[0:2000])
plt.plot(t[0:2000],output5[0:2000])
plt.legend(["input", "output"])
plt.xlabel("Time [s]")
plt.ylabel("Level [cm]")
plt.title("Two input system control using fuzzy t2 controler")
plt.grid()
plt.show()

plt.plot(t[0:2000],error5[0:2000])
plt.xlabel("Time [s]")
plt.ylabel("Level [cm]")
plt.title("Error diagram")
plt.grid()
plt.show()

plt.plot(t[0:2000],outputs5[0:2000])
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.title("Vent pipe position")
plt.grid()
plt.show()

plt.plot(t[0:2000],derivation[0:2000])
plt.xlabel("Time [s]")
plt.ylabel("Derivation of output")
plt.title("Derivation over time")
plt.grid()
plt.show()
