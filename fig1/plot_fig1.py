#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("../src/")
from mfmodeling import NARGP, SingleGP
# help(NARGP)


# ### Preparation of data set

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt

np.random.seed(24)

''' function definitions '''
def high(x):
    return (x-np.sqrt(2))*low(x)**2
    # return low(x)**2

def low(x):
    return np.sin(4.0*np.pi*x)

''' Define training and test points '''
dim = 1
s = 2
plot = 1
N1 = 50
N2 = np.array([15])

Nts = 400
Xtest = np.linspace(0,2, Nts)[:,None]
Exact= high(Xtest)
Low = low(Xtest)

X1 = np.linspace(0,2, N1)[:,None]
# perm = np.random.permutation(N1)
perm = np.random.permutation(int(N1*0.6))
X2 = X1[perm[0:N2[0]]]

Y1 = low(X1)
Y2 = high(X2)

plt.plot(Xtest,Exact,label="Target function")
plt.plot(Xtest,Low,"--")
plt.scatter(X2,Y2,label="High-fidelity data")
plt.scatter(X1,Y1,label="Low-fidelity data")
plt.legend()
plt.ylim(-1.5,1.5)
plt.show()

data_low = np.hstack([X1,Y1])
np.savetxt("low_fid.txt",data_low)
data_high = np.hstack([X2,Y2])
np.savetxt("high_fid.txt",data_high)


# ### Regression model by NARGP

# In[ ]:


data_list = [[X1,Y1], # Low-fidelity data set
             [X2,Y2]] # High-fidelity data set
model_nargp = NARGP(data_list=data_list)
model_nargp.optimize(optimize_restarts=10, max_iters=200)
mean, var = model_nargp.predict(Xtest)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(Xtest, Exact, 'b', label='Exact', linewidth = 2)
ax.plot(Xtest, mean, 'r--', label = 'NARGP prediction', linewidth = 2)
ax.fill_between(Xtest.ravel(), (mean-2.0*np.sqrt(var)).ravel(), (mean+2.0*np.sqrt(var)).ravel(), alpha=0.1, color='red')
ax.plot(X1, Y1,'g.', label="Low-fidelity data")
ax.plot(X2, Y2,'bo', label="High-fidelity data")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-2,1.5)
ax.legend()
plt.show()

mean0, var0 = model_nargp.predict(Xtest,ifidelity=0)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(Low, Exact, label = "Exact correlation")
ax.plot(mean0, mean, label = "Predicted correlation")
ax.set_xlabel(r"Low fidelity $y_0$")
ax.set_ylabel(r"High fidelity $y_1$")
ax.legend()
plt.show()


# ### Regression model by single GP

# In[ ]:


data = [X2,Y2] # Single GP using only high-fidelity data
model_singlegp = SingleGP(data = data)
model_singlegp.optimize(optimize_restarts=10, max_iters=200)
mean_singlegp, var_singlegp = model_singlegp.predict(Xtest)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot()
ax.plot(Xtest, Exact, 'b', label='Exact', linewidth = 2)
ax.plot(Xtest, mean_singlegp, 'r--', label = 'SingleGP prediction', linewidth = 2)
ax.fill_between(Xtest.ravel(), (mean_singlegp-2.0*np.sqrt(var_singlegp)).ravel(), (mean_singlegp+2.0*np.sqrt(var_singlegp)).ravel(), alpha=0.1, color='red')
ax.plot(X1, Y1,'g.', label="Low-fidelity data")
ax.plot(X2, Y2,'bo', label="High-fidelity data")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-2,1.5)
ax.legend()
plt.show()


# ### Plot Figure 1

# In[ ]:


plt.style.use('../nature_style.txt')

fig=plt.figure(figsize=(3.3,5),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
ax=fig.add_subplot(3,1,1)
ax.plot(X1,Y1,"g+",label="Low-fidelity data")
ax.plot(Xtest,Low,"g--",lw=1)
ax.plot(X2,Y2,"bo",markersize=2,label="High-fidelity data")
ax.plot(Xtest,Exact,"b",lw=1,label="Target function")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xlim(0,2)
ax.set_ylim(-2.2,1.2)
ax.legend(ncol=2)
ax.text(1.82,0.8,"(a)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

ax=fig.add_subplot(3,1,2)
# ax.plot(X1, Y1,'g.', label="Low-fidelity data")
ax.plot(X2,Y2,"bo",markersize=2,label="High-fidelity data")
ax.plot(Xtest,Exact,"b",lw=1,label="Target function")
ax.plot(Xtest, mean_singlegp, 'r--', lw=1,label = 'SingleGP pred.')
ax.fill_between(Xtest.ravel(), (mean_singlegp-2.0*np.sqrt(var_singlegp)).ravel(), (mean_singlegp+2.0*np.sqrt(var_singlegp)).ravel(), alpha=0.1, color='red')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xlim(0,2)
ax.set_ylim(-2.2,1.2)
ax.legend(ncol=2)
ax.text(1.82,0.8,"(b)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

ax = fig.add_subplot(3,1,3)
ax.plot(X1,Y1,"g+",label="Low-fidelity data")
ax.plot(X2,Y2,"bo",markersize=2,label="High-fidelity data")
ax.plot(Xtest,Exact,"b",lw=1,label="Target function")
ax.plot(Xtest, mean, 'r--', lw=1, label = 'NARGP pred.')
ax.fill_between(Xtest.ravel(), (mean-2.0*np.sqrt(var)).ravel(), (mean+2.0*np.sqrt(var)).ravel(), alpha=0.1, color='red')
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_xlim(0,2)
ax.set_ylim(-2.2,1.2)
ax.legend(ncol=2)
ax.text(1.82,0.8,"(c)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

fig.tight_layout()
plt.savefig("fig1.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# In[ ]:




