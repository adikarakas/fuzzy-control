# Type I controller for two-input system

import numpy as np
import matplotlib.pyplot as plt
import math as math

import pyit2fls
from pyit2fls import T1FS, T1FS_plot, T1Mamdani
from pyit2fls import IT2FS, IT2FLS, IT2FS_plot, trapezoid_mf, tri_mf, TR_plot, crisp, min_t_norm, max_s_norm, product_t_norm

domain = np.linspace(-2, 2, 1001)
domain2 = np.linspace(0, 1, 251)
domain3 = np.linspace(-1, 1, 501)

class2=T1Mamdani()
class2.add_input_variable("x_1")
class2.add_input_variable("x_2")
class2.add_output_variable("y_1")

bigneg1 = T1FS(domain,trapezoid_mf,[-15/5, -12/5, -9/5, -6/5, 1])
litneg1 = T1FS(domain, trapezoid_mf, [-8/5, -6/5, -2.5/5, -0.5/5, 1])
middle1 = T1FS(domain, tri_mf, [-1.5/5, 0/5, 1.5/5, 1])
litpos1 = T1FS(domain, trapezoid_mf, [0.5/5, 2.5/5, 6/5, 8/5, 1])
bigpos1 = T1FS(domain, trapezoid_mf, [6/5, 9/5, 12/5, 15/5, 1])
T1FS_plot(bigneg1, litneg1, middle1, litpos1, bigpos1, title="Input1: Error value", legends=["big negative", "little negative", "negligible", "little positive", "big positive"], xlabel="Domain", ylabel="Value")

neg1 = T1FS(domain3, trapezoid_mf, [-1.1, -1, -0.4, -0.01, 1])
nul1 = T1FS(domain3, tri_mf, [-0.02, 0, 0.02, 1])
poz1 = T1FS(domain3, trapezoid_mf, [0.01, 0.4, 1, 1.1, 1])
T1FS_plot(neg1, nul1, poz1, title="Input2: Value of derivation of the error", legends=["negative", "null", "positive"], xlabel="Domain", ylabel="Value")

fulcls11 = T1FS(domain2, trapezoid_mf, [-1, -0.4, 0, 0.175, 1])
parcls11 = T1FS(domain2, trapezoid_mf, [0.05, 0.175, 0.325, 0.45, 1])
mid11 = T1FS(domain2, tri_mf, [0.425, 0.5, 0.575, 1])
paropn11 = T1FS(domain2, trapezoid_mf, [0.55, 0.672, 0.825, 0.95, 1])
fulopn11 = T1FS(domain2, trapezoid_mf, [0.825, 1, 1.4, 2, 1])
T1FS_plot(fulcls11, parcls11, mid11, paropn11, fulopn11, title="Output: Vent pipe position", legends=["fully closed", "partially closed", "middle", "partially opened", "fully opened"], xlabel="Domain", ylabel="Value")

class2.add_rule([("x_1", middle1), ("x_2", neg1)], [("y_1", mid11)])
class2.add_rule([("x_1", middle1), ("x_2", nul1)], [("y_1", mid11)])
class2.add_rule([("x_1", middle1), ("x_2", poz1)], [("y_1", mid11)])
class2.add_rule([("x_1", litneg1), ("x_2", neg1)], [("y_1", parcls11)])
class2.add_rule([("x_1", litneg1), ("x_2", nul1)], [("y_1", parcls11)])
class2.add_rule([("x_1", litneg1), ("x_2", poz1)], [("y_1", mid11)])
class2.add_rule([("x_1", litpos1), ("x_2", neg1)], [("y_1", mid11)])
class2.add_rule([("x_1", litpos1), ("x_2", nul1)], [("y_1", paropn11)])
class2.add_rule([("x_1", litpos1), ("x_2", poz1)], [("y_1", paropn11)])
class2.add_rule([("x_1", bigneg1), ("x_2", neg1)], [("y_1", fulcls11)])
class2.add_rule([("x_1", bigneg1), ("x_2", nul1)], [("y_1", fulcls11)])
class2.add_rule([("x_1", bigneg1), ("x_2", poz1)], [("y_1", parcls11)])
class2.add_rule([("x_1", bigpos1), ("x_2", neg1)], [("y_1", paropn11)])
class2.add_rule([("x_1", bigpos1), ("x_2", nul1)], [("y_1", fulopn11)])
class2.add_rule([("x_1", bigpos1), ("x_2", poz1)], [("y_1", fulopn11)])

domainn=np.linspace(-2, 2, 101)
domainn2=np.linspace(-1, 1, 101)

Z=np.ones((domainn.shape[0], domainn2.shape[0]))
for i, x1 in zip(range(len(domainn)),domainn):
	for j, x2 in zip(range(len(domainn2)),domainn2):
		it2out, tr=class2.evaluate({"x_1":x1, "x_2":x2})
		Z[i][j]=tr["y_1"]

X1, X2 = np.meshgrid(domainn, domainn2)

fig = plt.figure()
ax = fig.gca(projection="3d")
surf = ax.plot_surface(X1, X2, Z)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


g=9.81
lb=0
ub=2*1
area=1
S=0.05
lower=0.5*np.ones(200)
upper=1.5*np.ones(200)
input=np.hstack((upper, lower, upper, lower, upper, lower, upper, lower, upper, lower))
error2=np.zeros(2000)
output2=np.zeros(2001)
output2[0]=0.5
derivation=0
flow_out=0.1565
first=0.5
i=-1
previous=0.5
for t in range(1,2000,1):
  t=t/20
  i=i+1
  error2[i]=input[i]-output2[i]
  it2out, tr=class2.evaluate({"x_1":error2[i], "x_2":derivation})
  fuzzy_output=tr["y_1"]
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
  output2[i+1]=limited/area
  derivation=(output2[i+1]-output2[i])/0.1
  if (derivation<-1):
    derivation=-1
  elif (derivation>1):
    derivation=1
  velocity=math.sqrt(2*g*output2[i+1])
  flow_out=velocity*S

t=np.linspace(0,200,2001)

plt.plot(t[0:2000],input[0:2000])
plt.plot(t[0:2000],output2[0:2000])
plt.legend(["input", "output"])
plt.xlabel("time [s]")
plt.ylabel("level [cm]")
plt.title("System control using fuzzy t2 controler")
plt.grid()
plt.show()

plt.plot(t[0:2000],error2[0:2000])
plt.legend(["input", "output"])
plt.xlabel("time [s]")
plt.ylabel("level [cm]")
plt.title("Error diagram")
plt.grid()
plt.show()
