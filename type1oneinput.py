# Type I controller for one-input system

import numpy as np
import matplotlib.pyplot as plt
import math as math

import pyit2fls
from pyit2fls import T1FS, T1FS_plot, T1Mamdani
from pyit2fls import IT2FS, IT2FLS, IT2FS_plot, trapezoid_mf, tri_mf, TR_plot, crisp, min_t_norm, max_s_norm, product_t_norm

domain = np.linspace(-2, 2, 1001)
domain2 = np.linspace(0, 1, 251)
domain3 = np.linspace(-1, 1, 501)

class1=T1Mamdani()

bigneg1 = T1FS(domain,trapezoid_mf,[-15/5, -12/5, -9/5, -6/5, 1])
litneg1 = T1FS(domain, trapezoid_mf, [-8/5, -6/5, -2.5/5, -0.5/5, 1])
middle1 = T1FS(domain, tri_mf, [-1.5/5, 0/5, 1.5/5, 1])
litpos1 = T1FS(domain, trapezoid_mf, [0.5/5, 2.5/5, 6/5, 8/5, 1])
bigpos1 = T1FS(domain, trapezoid_mf, [6/5, 9/5, 12/5, 15/5, 1])
class1.add_input_variable("x")
T1FS_plot(bigneg1, litneg1, middle1, litpos1, bigpos1, title="Input: Error value", legends=["big negative", "little negative", "negligible", "little positive", "big positive"], xlabel="Domain", ylabel="Value")

fulcls1 = T1FS(domain2, trapezoid_mf, [-1, 0, 0.25, 0.4, 1])
mid1 = T1FS(domain2, tri_mf, [0.35, 0.5, 0.65, 1])
fulopn1 = T1FS(domain2, trapezoid_mf, [0.6, 0.75, 1, 2, 1])
class1.add_output_variable("y")
T1FS_plot(fulcls1, mid1, fulopn1, title="Output: Vent pipe position", legends=["fully closed", "middle", "fully opened"], xlabel="Domain", ylabel="Value")

class1.add_rule([("x", middle1)], [("y", mid1)])
class1.add_rule([("x", litneg1)], [("y", fulcls1)])
class1.add_rule([("x", litpos1)], [("y", fulopn1)])
class1.add_rule([("x", bigneg1)], [("y", fulcls1)])
class1.add_rule([("x", bigpos1)], [("y", fulopn1)])

Z=np.zeros(domain.shape)
for i, x1 in zip(range(len(domain)),domain):
	it2out, tr =class1.evaluate({"x":x1})
	Z[i]=tr["y"]

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
error1=np.zeros(2000)
output1=np.zeros(2001)
output1[0]=0.5
flow_out=0.1565
first=0.5
i=-1
previous=0.5
for t in range(1,2000,1):
  t=t/20
  i=i+1
  error1[i]=input[i]-output1[i]
  it2out, tr=class1.evaluate({"x":error1[i]})
  fuzzy_output=tr["y"]
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
  output1[i+1]=limited/area
  velocity=math.sqrt(2*g*output1[i+1])
  flow_out=velocity*S

t=np.linspace(0,200,2001)

plt.plot(t[0:2000],input[0:2000])
plt.plot(t[0:2000],output1[0:2000])
plt.legend(["input", "output"])
plt.xlabel("Time [s]")
plt.ylabel("Level [cm]")
plt.title("System control using fuzzy t1 controler")
plt.grid()
plt.show()

plt.plot(t[0:2000],error1[0:2000])
plt.legend(["input", "output"])
plt.xlabel("Time [s]")
plt.ylabel("Level [cm]")
plt.title("Error diagram")
plt.grid()
plt.show()
