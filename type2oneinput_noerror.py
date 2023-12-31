# Type II error-eliminating controller for one-input system

import numpy as np
import matplotlib.pyplot as plt
import math as math

import pyit2fls
from pyit2fls import T1FS, T1FS_plot, T1Mamdani
from pyit2fls import IT2FS, IT2FLS, IT2FS_plot, trapezoid_mf, tri_mf, TR_plot, crisp, min_t_norm, max_s_norm, product_t_norm

domain = np.linspace(-2, 2, 1001)
domain2 = np.linspace(0, 1, 251)
domain3 = np.linspace(-1, 1, 501)

class_1 = IT2FLS()

litneg_1 = IT2FS(domain, trapezoid_mf, [-1.7, -1.5, -0.2, 0.002, 1], trapezoid_mf, [-1.5, -1.3, -0.4, -0.2, 0.7])
litpos_1 = IT2FS(domain, trapezoid_mf, [-0.002, 0.2, 1.5, 1.7, 1], trapezoid_mf, [0.2, 0.4, 1.3, 1.5, 0.7])
bigpos_1 = IT2FS(domain, trapezoid_mf, [1.1, 1.6, 2.1, 3.1, 1], trapezoid_mf, [1.3, 1.8, 2.3, 3.3, 0.7])
bigneg_1 = IT2FS(domain, trapezoid_mf, [-3.1, -2.1, -1.6, -1.1, 1], trapezoid_mf, [-3.3, -2.3, -1.8, -1.3, 0.7])

IT2FS_plot(bigneg_1, litneg_1, litpos_1, bigpos_1, title="Input: Error value", legends=["big negative", "little negative", "little positive", "big positive"], xlabel="Domain", ylabel="Value")
class_1.add_input_variable("x1")

fulcls_1 = IT2FS(domain2, trapezoid_mf, [-0.35, -0.25, 0.4, 0.5, 1], trapezoid_mf, [-0.25, -0.15, 0.3, 0.4, 0.7])
fulopn_1 = IT2FS(domain2, trapezoid_mf, [0.5, 0.6, 1.25, 1.35, 1], trapezoid_mf, [0.6, 0.7, 1.15, 1.25, 0.7])

IT2FS_plot(fulcls_1, fulopn_1, title="Output: Vent pipe position", legends=["fully closed", "fully opened"], xlabel="Domain", ylabel="Value")
class_1.add_output_variable("y1")

class_1.add_rule([("x1", litneg_1)], [("y1", fulcls_1)])
class_1.add_rule([("x1", litpos_1)], [("y1", fulopn_1)])
class_1.add_rule([("x1", bigneg_1)], [("y1", fulcls_1)])
class_1.add_rule([("x1", bigpos_1)], [("y1", fulopn_1)])

Z=np.zeros(domain.shape)
for i, x1 in zip(range(len(domain)),domain):
	it2out, tr =class_1.evaluate({"x1":x1}, min_t_norm, max_s_norm, domain)
	Z[i]=crisp(tr["y1"])

plt.plot(domain,Z)
plt.xlim([-2, 2])
plt.ylim([0, 1])
plt.xlabel("Input value")
plt.ylabel("Output value")
plt.title("Rule base surface")
plt.grid()
plt.show()

g=9.81
lb=0
ub=2*1
area=1
S=0.05
lower=0.5*np.ones(200)
upper=1.5*np.ones(200)
input=np.hstack((upper, lower, upper, lower, upper, lower, upper, lower, upper, lower))
output_1=np.zeros(2001)
error_1=np.zeros(2000)
outputs_1=np.zeros(2000)
output_1[0]=0.5
flow_out=0.1565
first=0.5
i=-1
previous=0.5
for t in range(1,2000,1):
  t=t/20
  i=i+1
  error_1[i]=input[i]-output_1[i]
  it2out, tr=class_1.evaluate({"x1":error_1[i]}, min_t_norm, max_s_norm, domain, method="Centroid")
  fuzzy_output=crisp(tr["y1"])
  outputs_1[i]=fuzzy_output
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
  output_1[i+1]=limited/area
  velocity=math.sqrt(2*g*output_1[i+1])
  flow_out=velocity*S

t=np.linspace(0,200,2001)

plt.plot(t[0:2000],input[0:2000])
plt.plot(t[0:2000],output_1[0:2000])
plt.legend(["input", "output"])
plt.xlabel("Time [s]")
plt.ylabel("Level [cm]")
plt.title("System control using fuzzy t2 controler for error elimination")
plt.grid()
plt.show()

plt.plot(t[0:2000],error_1[0:2000])
plt.xlabel("Time [s]")
plt.ylabel("Level [cm]")
plt.title("Error diagram")
plt.grid()
plt.show()

plt.plot(t[0:2000],outputs_1[0:2000])
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.title("Vent pipe position over time")
plt.grid()
plt.show()
