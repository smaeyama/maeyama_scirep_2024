#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("../src/")
from mfmodeling import NARGP, SingleGP
# help(NARGP)


# ### Read dataset

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

df = pd.read_csv("./data/JET_data_Narita2023CPP_fig1.csv")
df = df[df["D_exp"]>0].copy()
print(df.columns)

df_train = df.sample(df.shape[0]//3, random_state=21)
df_test = df.drop(df_train.index)
print(df.shape)
print(df_train.shape)
print(df_test.shape)


# In[ ]:


fig = plt.figure(figsize=(12,6))
fig.suptitle("D_exp")
nl=3; nc=4; i=0
for l in range(nl):
    for c in range(nc):
        ax = fig.add_subplot(nl,nc,1+i)
        ax.plot(df.iloc[:,i],df["D_exp"],"x",label="All data")
        ax.plot(df_train.iloc[:,i],df_train["D_exp"],".",label="Train")
        ax.plot(df_test.iloc[:,i],df_test["D_exp"],".",label="Test")
        ax.set_xlabel(df_train.columns[i])
        # ax.set_xscale("log")
        ax.set_yscale("log")
        if i==0:
            ax.legend()
        i=i+1
fig.tight_layout()
plt.show()


# ### Quasi-linear model of Eq. (6) in Narita, Contribut. Plasma Phys.

# In[ ]:


def D_model_eq6(param,df):
    c0, alpha, beta = param
    Z_i = 1
    nu_ii = Z_i**4 * df["ni/ne"] * df["Te/Ti"]**2 * df["nu_ee"]
    nu_ii_star = nu_ii * df["q"] / df["epsilon"]**1.5
    tau_ii = 3*np.sqrt(np.pi)/(4*nu_ii)
    K_RH = 1/(1+1.6*df["q"]**2/np.sqrt(df["epsilon"]))
    tau_r_bar = np.sqrt(np.exp(-df["q"]**2)**2+K_RH**2*(tau_ii*df["epsilon"]/0.67)**2)
    mixing = np.array(df["gamma"]/df["ktheta"]**2)
    D_model = c0*mixing**alpha*(0.1*tau_r_bar*mixing**0.5)**beta
    return D_model

def fit_eq6(param,df,sol):
    # return np.abs(sol - D_model_eq6(param,df)) # Linear fit
    return np.abs(np.log(sol) - np.log(D_model_eq6(param,df))) # Log fit

param_eq6 = [1.55, 0.75, -1.2]
optimize_result = optimize.least_squares(fit_eq6, param_eq6, args=(df_train,df_train["D_exp"]))
param_eq6 = optimize_result.x
print(param_eq6)


# In[ ]:


ans=np.array(df_test["D_exp"])
pred=D_model_eq6(param_eq6,df_test)
MSLE_eq6 = ((np.log(1+pred)-np.log(1+ans))**2).mean() # Linear fit
# R2_eq6 = 1 - np.sum((ans-pred)**2)/np.sum((ans-ans.mean())**2)
R2_eq6 = 1 - np.sum((np.log(ans)-np.log(pred))**2)/np.sum((np.log(ans)-np.log(ans).mean())**2)
print("MSLE=",MSLE_eq6, ", R^2=",R2_eq6)

fig=plt.figure()
ax = fig.add_subplot()
D_min = 0.02 #df_all["D_exp"].min()
D_max = 40 #df_all["D_exp"].max()
ax.plot([D_min,D_max],[D_min,D_max],"--",color="k",lw=1)
ax.plot(df_train["D_exp"],D_model_eq6(param_eq6,df_train),"g+",label="Train - Eq.(6)")
ax.plot(df_test["D_exp"],D_model_eq6(param_eq6,df_test),"r.",label="Test - Eq.(6)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(D_min,D_max)
ax.set_ylim(D_min,D_max)
ax.set_aspect("equal")
ax.set_xlabel("D_exp")
ax.set_ylabel("D_model")
ax.legend()
plt.show()


# ### Single GP

# In[ ]:


# data_list = [[np.array(df_train)[:,:12],np.array(df_train["D_exp"])[:,np.newaxis]]] # Linear fit
data_list = [[np.array(df_train)[:,:12],np.log(np.array(df_train["D_exp"])[:,np.newaxis])]] # Log fit
print(data_list[0][0].shape, data_list[0][1].shape)
model_singlegp = NARGP(data_list = data_list)             
model_singlegp.optimize(optimize_restarts=20, max_iters=400)


# In[ ]:


mean_train_1, var_train_1 = model_singlegp.predict(np.array(df_train)[:,:12])
mean_test_1, var_test_1 = model_singlegp.predict(np.array(df_test)[:,:12])

# ### Linear fit ###
# ans=np.array(df_test["D_exp"])
# pred=mean_test.ravel()
### Log fit ###
ans=np.array(df_test["D_exp"])
pred=np.exp(mean_test_1.ravel())

MSLE_singlegp = ((np.log(1+pred)-np.log(1+ans))**2).mean() # Linear fit
# R2 = 1 - np.sum((ans-pred)**2)/np.sum((ans-ans.mean())**2)
R2_singlegp = 1 - np.sum((np.log(ans)-np.log(pred))**2)/np.sum((np.log(ans)-np.log(ans).mean())**2)
print("MSLE=",MSLE_singlegp, ", R^2=",R2_singlegp)

fig=plt.figure()
ax = fig.add_subplot()
D_min = 0.02 #df_all["D_exp"].min()
D_max = 40 #df_all["D_exp"].max()
ax.plot([D_min,D_max],[D_min,D_max],"--",color="k",lw=1)
# ax.plot(df_train["D_exp"],mean_train,"o",label="Train") # Linear fit
# ax.plot(df_test["D_exp"],mean_test,"o",label="Test") # Linear fit
ax.plot(df_train["D_exp"],np.exp(mean_train_1),"g+",label="Train") # Log fit
ax.plot(df_test["D_exp"],np.exp(mean_test_1),"r.",label="Test") # Log fit
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(D_min,D_max)
ax.set_ylim(D_min,D_max)
ax.set_aspect("equal")
ax.set_xlabel("D_exp")
ax.set_ylabel("D_model")
ax.legend()
plt.show()


# ### Multi-fidelity regression by NARGP
# Low-fidelity data: (linearly most unstable wavenumber, growthrate)
# High-fidelity data: D_exp

# In[ ]:


# data_list = [[np.array(df)[:,:12],np.array(df)[:,17:19]],,
#              [np.array(df_train)[:,:12],np.array(df_train["D_exp"])[:,np.newaxis]]] # Linear fit
data_list = [[np.array(df)[:,:12],np.array(df)[:,17:19]],
             [np.array(df_train)[:,:12],np.log(np.array(df_train["D_exp"])[:,np.newaxis])]] # Log fit
print(data_list[0][0].shape, data_list[0][1].shape)
print(data_list[1][0].shape, data_list[1][1].shape)
model_nargp = NARGP(data_list=data_list)             
model_nargp.optimize(optimize_restarts=20, max_iters=400)


# In[ ]:


mean_train, var_train = model_nargp.predict(np.array(df_train)[:,:12])
mean_test, var_test = model_nargp.predict(np.array(df_test)[:,:12])
# mean_train = mean_train + np.log(np.array(D_model_eq6(param_eq6,df_train))[:,np.newaxis])
# mean_test = mean_test + np.log(np.array(D_model_eq6(param_eq6,df_test))[:,np.newaxis])

# ### Linear fit ###
# ans=np.array(df_test["D_exp"])
# pred=mean_test.ravel().copy()
### Log fit ###
ans=np.array(df_test["D_exp"])
pred=np.exp(mean_test.ravel())

MSLE=((np.log(1+pred)-np.log(1+ans))**2).mean()
# R2 = 1 - np.sum((ans-pred)**2)/np.sum((ans-ans.mean())**2)
R2 = 1 - np.sum((np.log(ans)-np.log(pred))**2)/np.sum((np.log(ans)-np.log(ans).mean())**2)
print("MSLE=",MSLE, ", R^2=",R2)

fig=plt.figure()
ax = fig.add_subplot()
D_min = 0.02 #df_all["D_exp"].min()
D_max = 40 #df_all["D_exp"].max()
ax.plot([D_min,D_max],[D_min,D_max],"--",color="k",lw=1)
# ax.plot(df_train["D_exp"],mean_train,"o",label="Train") # Linear fit
# ax.plot(df_test["D_exp"],mean_test,"o",label="Test") # Linear fit
ax.plot(df_train["D_exp"],np.exp(mean_train),"o",label="Train") # Log fit
ax.plot(df_test["D_exp"],np.exp(mean_test),"o",label="Test") # Log fit
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(D_min,D_max)
ax.set_ylim(D_min,D_max)
ax.set_aspect("equal")
ax.set_xlabel("D_exp")
ax.set_ylabel("D_model")
ax.legend()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(12,6))
fig.suptitle("D_exp")
nl=3; nc=4; i=0
for l in range(nl):
    for c in range(nc):
        ax = fig.add_subplot(nl,nc,1+i)
        ax.plot(df_train.iloc[:,i],df_train["D_exp"],"cx",label="Data(Train)")
        # ax.plot(df_test.iloc[:,i],df_test["D_exp"],"mx",label="Data(Test)")
        ax.plot(df_train.iloc[:,i],np.exp(mean_train),"b.",label="Pred(Train)")
        # ax.plot(df_test.iloc[:,i],np.exp(mean_test),"r.",label="Pred(Test)")
        ax.set_xlabel(df_train.columns[i])
        # ax.set_xscale("log")
        ax.set_yscale("log")
        if i==0:
            ax.legend()
        i=i+1
fig.tight_layout()
plt.show()

fig = plt.figure(figsize=(12,6))
fig.suptitle("D_exp")
nl=3; nc=4; i=0
for l in range(nl):
    for c in range(nc):
        ax = fig.add_subplot(nl,nc,1+i)
        # ax.plot(df_train.iloc[:,i],df_train["D_exp"],"cx",label="Data(Train)")
        ax.plot(df_test.iloc[:,i],df_test["D_exp"],"mx",label="Data(Test)")
        # ax.plot(df_train.iloc[:,i],np.exp(mean_train),"b.",label="Pred(Train)")
        ax.plot(df_test.iloc[:,i],np.exp(mean_test),"r.",label="Pred(Test)")
        ax.set_xlabel(df_train.columns[i])
        # ax.set_xscale("log")
        ax.set_yscale("log")
        if i==0:
            ax.legend()
        i=i+1
fig.tight_layout()
plt.show()


# In[ ]:





# ### Plot Figure 4

# In[ ]:


plt.style.use('../nature_style.txt')

fig=plt.figure(figsize=(5.9,5),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)

xlabels = [r"$R/L_n$", r"$R/L_{T_e}$", r"$R/L_{T_i}$", r"$n_i/n_e$", r"$T_e/T_i$", r"$\beta$", r"$\nu_{ee}$", r"$q_0$", r"$\hat{s}$", r"$\epsilon$", r"$\kappa$", r"$\delta$"]
text = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)"]
# fig.suptitle("D_exp")
nl=4; nc=3; i=0
D_min = 0.01 #df["D_exp"].min()
D_max = 20 #df["D_exp"].max()
for l in range(nl):
    for c in range(nc):
        ax = fig.add_subplot(nl,nc,1+i)
        # ax.plot(df.iloc[:,i],df["D_exp"],"x",label="All data")
        ax.plot(df_train.iloc[:,i],df_train["D_exp"],"g+",lw=1,label="Train")
        ax.plot(df_test.iloc[:,i],df_test["D_exp"],"bx",lw=1,label="Test")
        ax.set_xlabel(xlabels[i])
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(D_min,D_max)
        if i==7 or i==9 or i==10:
            ax.text(0.05, 0.82,text[i],color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8, transform=ax.transAxes)
        else:
            ax.text(0.835, 0.82,text[i],color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8, transform=ax.transAxes)
        if c==0:
            ax.set_ylabel(r"$D_\mathrm{exp}$")
        if i==0:
            ax.legend()
        i=i+1
fig.tight_layout()
plt.savefig("fig4.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# ### Plot Figure 5

# In[ ]:


plt.style.use('../nature_style.txt')

fig=plt.figure(figsize=(5.9,3),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)

ax = fig.add_subplot(1,3,1)
D_min = 0.01 #df["D_exp"].min()
D_max = 20 #df["D_exp"].max()
ax.plot([D_min,D_max],[D_min,D_max],"--",color="k",lw=0.5)
ax.plot(D_model_eq6(param_eq6,df_train),df_train["D_exp"],"+",label="Train")
ax.plot(D_model_eq6(param_eq6,df_test),df_test["D_exp"],".",label="Test")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(D_min,D_max)
ax.set_ylim(D_min,D_max)
ax.set_aspect("equal")
ax.set_aspect("equal")
ax.set_xlabel(r"Prediction $D_\mathrm{QL}$")
ax.set_ylabel(r"Experimental $D_\mathrm{exp}$")
ax.legend(loc="lower right")
ax.text(0.015,9,"(a)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

ax = fig.add_subplot(1,3,2)
D_min = 0.01 #df["D_exp"].min()
D_max = 20 #df["D_exp"].max()
ax.plot([D_min,D_max],[D_min,D_max],"--",color="k",lw=0.5)
ax.plot(np.exp(mean_train_1),df_train["D_exp"],"+",label="Train")
ax.plot(np.exp(mean_test_1),df_test["D_exp"],".",label="Test")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(D_min,D_max)
ax.set_ylim(D_min,D_max)
ax.set_aspect("equal")
ax.set_aspect("equal")
ax.set_xlabel(r"Prediction $D_\mathrm{SingleGP}$")
ax.set_ylabel(r"Experimental $D_\mathrm{exp}$")
ax.legend(loc="lower right")
ax.text(0.015,9,"(b)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

ax = fig.add_subplot(1,3,3)
D_min = 0.01 #df["D_exp"].min()
D_max = 20 #df["D_exp"].max()
ax.plot([D_min,D_max],[D_min,D_max],"--",color="k",lw=0.5)
ax.plot(np.exp(mean_train),df_train["D_exp"],"+",label="Train")
ax.plot(np.exp(mean_test),df_test["D_exp"],".",label="Test")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(D_min,D_max)
ax.set_ylim(D_min,D_max)
ax.set_aspect("equal")
ax.set_aspect("equal")
ax.set_xlabel(r"Prediction $D_\mathrm{NARGP}$")
ax.set_ylabel(r"Experimental $D_\mathrm{exp}$")
ax.legend(loc="lower right")
ax.text(0.015,9,"(c)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

plt.tight_layout()
plt.savefig("fig5.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# In[ ]:




