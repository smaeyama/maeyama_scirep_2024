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

df = pd.read_csv("./data/LHD_df1.csv")
label=df.columns.values
print(df.shape)
print(label[:9])
print(label[9:14])
print(label[14:])

df_train = df[1::3]
df_test = df.drop(df_train.index)
print(df_train.shape, df_test.shape)


# In[ ]:


name="Gamma_nlnr"
qslname="Gamma_qslmodel"
fig = plt.figure(figsize=(10,8))
fig.suptitle(name)
for i in range(14):
    ax = fig.add_subplot(4,4,i+1)
    # ax.plot(df.iloc[:,i],df.loc[:,name],"s")
    ax.plot(df_train.iloc[:,i],df_train.loc[:,name],"b+",label="Train")
    ax.plot(df_test.iloc[:,i],df_test.loc[:,name],"gx",label="Test")
    # ax.plot(df.iloc[:,i],df.loc[:,qslname],"r.",label="Q.L.")
    ax.set_xlabel(label[i])
    if i == 0:
        ax.legend()
fig.tight_layout()
plt.show()


# ### Regression by NARGP
# Low-fidelity data: Quasi-linear transport model
# High-fidelity data: Nonlinear simulation result of turbulent flux

# In[ ]:


data_list = [[np.array(df)[:,1:9],np.array(df[qslname])[:,np.newaxis]],
             [np.array(df_train)[:,1:9],np.array(df_train[name])[:,np.newaxis]-np.array(df_train[qslname])[:,np.newaxis]]]
print(data_list[0][0].shape, data_list[0][1].shape)
print(data_list[1][0].shape, data_list[1][1].shape)
model_nargp = NARGP(data_list=data_list)             
model_nargp.optimize(optimize_restarts=40, max_iters=400)


# In[ ]:


mean_train, var_train = model_nargp.predict(np.array(df_train)[:,1:9])
mean_qsl_train, _ = model_nargp.predict(np.array(df_train)[:,1:9],ifidelity=0)
mean_train = mean_qsl_train + mean_train
mean_test, var_test = model_nargp.predict(np.array(df_test)[:,1:9])
mean_qsl_test, _ = model_nargp.predict(np.array(df_test)[:,1:9],ifidelity=0)
mean_test = mean_qsl_test + mean_test

ans=np.array(df_test[name])
pred=mean_test.ravel()

MSLE=((np.log(1+pred)-np.log(1+ans))**2).mean()
R2 = 1 - np.sum((ans-pred)**2)/np.sum((ans-ans.mean())**2)
print("MSLE=",MSLE, ", R^2=",R2)
qsl=np.array(df_test[qslname])
R2_qsl = 1 - np.sum((ans-qsl)**2)/np.sum((ans-ans.mean())**2)
print("R_{qsl}^2=",R2_qsl)

fig=plt.figure()
ax = fig.add_subplot()
xmin = df[name].min()
xmax = df[name].max()
ax.plot([xmin,xmax],[xmin,xmax],"--",color="k",lw=1)
ax.plot(df_train[name],mean_train,"bo",label="Train")
ax.plot(df_test[name],mean_test,"go",label="Test")
# ax.plot(df_train[name],mean_qsl_train,"bo",label="Train")
# ax.plot(df_test[name],mean_qsl_test,"go",label="Test")
ax.plot(df_train[name],df_train[qslname],"bx",label="Q.L. model")
ax.plot(df_test[name],df_test[qslname],"gx",label="Q.L. model")
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(xmin,xmax)
ax.set_aspect("equal")
ax.set_xlabel(name)
ax.set_ylabel("NARGP")
ax.legend()
plt.show()


# ### Plot Figure 6

# In[ ]:


print(label)


# In[ ]:


plt.style.use('../nature_style.txt')

data_p = np.loadtxt("./data/LHDplasma1.txt",skiprows=3)
data_q = np.loadtxt("./data/LHDq1.txt",skiprows=3)

fig=plt.figure(figsize=(5.6,2.4),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
ax=fig.add_subplot(1,2,1)
ax.plot(df["rho"],df['n(10^19m-3)'],"x",c="C00",markersize=3,label=r"$n$ [$10^{19}\mathrm{m}^{-3}$]")
ax.plot(data_p[:,0],data_p[:,1],"-",lw=0.6,c="C00")
ax.plot(df["rho"],df['Te(keV)'],"+",c="C01",markersize=4,label=r"$T_e$ [keV]")
ax.plot(data_p[:,0],data_p[:,5],"-",lw=0.6,c="C01")
ax.plot(df["rho"],df['Ti(keV)'],".",c="C02",markersize=4,label=r"$T_i$ [keV]")
ax.plot(data_p[:,0],data_p[:,9],"-",lw=0.6,c="C02")
ax.plot(df["rho"],df['q_0'],"s",c="C03",markersize=2,label=r"$q$")
ax.plot(data_q[:,0],data_q[:,1],"-",lw=0.6,c="C03")
ax.set_xlabel(r"Minor radius $r/a$")
ax.set_ylabel(r"Plasma profiles")
ax.set_xlim(0.44,0.81)
ax.set_ylim(0,3.1)
ax.legend(ncol=2,loc="lower center")
ax.text(0.9, 0.91,"(a)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8, transform=ax.transAxes)

ax=fig.add_subplot(1,2,2)
ax.plot(df["rho"],df[qslname],"b+",markersize=4,label=r"Low-fidelity data $\Gamma_\mathrm{QL}$")
ax.plot(df_train["rho"],df_train[name],"gx",markersize=3,label=r"High-fidelity data $\Gamma_\mathrm{NL}$ (Train)")
# ax=fig.add_subplot(3,1,2)
# # ax.plot(X1, Y1,'g.', label="Low-fidelity data")
# ax.plot(X2,Y2,"bo",markersize=2,label="High-fidelity data")
# ax.plot(Xtest,Exact,"b",lw=1,label="Target function")
# ax.plot(Xtest, mean_singlegp, 'r--', lw=1,label = 'SingleGP pred.')
# ax.fill_between(Xtest.ravel(), (mean_singlegp-2.0*np.sqrt(var_singlegp)).ravel(), (mean_singlegp+2.0*np.sqrt(var_singlegp)).ravel(), alpha=0.1, color='red')
# ax.set_xlabel(r"$x$")
ax.set_xlabel(r"Minor radius $r/a$")
ax.set_ylabel(r"Turbulent transport fluxes")
ax.set_xlim(0.44,0.81)
ax.set_ylim(None,None)
ax.legend(loc="lower left")
ax.text(0.9, 0.91,"(b)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8, transform=ax.transAxes)


fig.tight_layout()
plt.savefig("fig6.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# ### Plot Figure 7

# In[ ]:


plt.style.use('../nature_style.txt')

fig=plt.figure(figsize=(5.35,2.4),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
gs=fig.add_gridspec(1,2,width_ratios=(1,1.23))
ax=fig.add_subplot(gs[0])
xmin = np.min([df[name].min(),df[qslname].min(),mean_train.min()])
xmax = np.max([df[name].max(),df[qslname].max(),mean_train.max()])
ax.plot([xmin,xmax],[xmin,xmax],"--",color="k",lw=1)
ax.plot(mean_train,df_train[name],"o",c="b",markersize=3,label="NARGP (Train)")
ax.plot(mean_test,df_test[name],"s",c="r",markersize=3,label="NARGP (Test)")
ax.plot(df_train[qslname],df_train[name],"+",c="C00",markersize=4,label="Q.L. (Train)")
ax.plot(df_test[qslname],df_test[name],"x",c="C01",markersize=3,label="Q.L. (Test)")
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(xmin,xmax)
ax.set_aspect("equal")
ax.set_xlabel("Prediction")
ax.set_ylabel("Nonlinear turbulent flux $\Gamma_\mathrm{NL}$")
ax.legend(loc="upper left",bbox_to_anchor=(0,0.92))
ax.text(0.035, 0.91,"(a)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8, transform=ax.transAxes)

mean, var = model_nargp.predict(np.array(df)[:,1:9])
mean_qsl, var_qsl = model_nargp.predict(np.array(df)[:,1:9],ifidelity=0)
mean = mean_qsl + mean
var = var_qsl + var # Estimation by assuming independent variance
# print(mean,df[name]-df[qslname])

ax=fig.add_subplot(gs[1])
ax.plot(df["rho"],df[qslname],"b+",markersize=4,label=r"Low-fidelity data $\Gamma_\mathrm{QL}$")
# ax.plot(df["rho"],df[name],"gx-",markersize=3,label=r"High-fidelity data $\Gamma_\mathrm{NL}$")
ax.plot(df["rho"],df[name], color="g", linestyle=(0, (3, 9)))
ax.plot(df["rho"],df[name], color="orange", linestyle=(6, (3, 9)))
ax.plot(df_train["rho"],df_train[name],"gx",markersize=3,label=r"High-fidelity data $\Gamma_\mathrm{NL}$ (Train)")
ax.plot(df_test["rho"],df_test[name],"^",color="orange",markersize=3,label=r"High-fidelity data $\Gamma_\mathrm{NL}$ (Test)")
ax.plot(df["rho"],mean,"ro-",markersize=3,label="NARGP pred. $\Gamma_\mathrm{NARGP}$")
ax.fill_between(df["rho"], (mean-2.0*np.sqrt(var)).ravel(), (mean+2.0*np.sqrt(var)).ravel(), alpha=0.1, color='red')
ax.set_xlabel(r"Minor radius $r/a$")
ax.set_ylabel(r"Turbulent transport fluxes")
ax.set_xlim(0.44,0.81)
ax.set_ylim(None,None)
ax.legend(loc="lower left")
ax.text(0.9, 0.91,"(b)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8, transform=ax.transAxes)

fig.tight_layout()
plt.savefig("fig7.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# In[ ]:


fig=plt.figure(figsize=(2.9,4.8),dpi=200) # figsize=(width,height(inch)),dpi(dots per inch)
gs=fig.add_gridspec(2,1,height_ratios=(1.5,1))
ax=fig.add_subplot(gs[1])
ax.plot(df["rho"],df[qslname],"b+",markersize=4,label=r"Low-fidelity data $\Gamma_\mathrm{QL}$")
ax.plot(df["rho"],df[name],"gx-",markersize=3,label=r"High-fidelity data $\Gamma_\mathrm{NL}$")
ax.plot(df["rho"],mean,"ro-",markersize=3,label="NARGP pred. $\Gamma_\mathrm{NARGP}$")
ax.plot(df["rho"],mean_qsl,"o-",color="orange",markersize=3,label="1st stage of NARGP $\Gamma_\mathrm{QL,NARGP}$")
ax.fill_between(df["rho"], (mean-2.0*np.sqrt(var)).ravel(), (mean+2.0*np.sqrt(var)).ravel(), alpha=0.1, color='red')
ax.set_xlabel(r"Minor radius $r/a$")
ax.set_ylabel(r"Turbulent transport fluxes")
ax.set_xlim(0.44,0.81)
ax.set_ylim(None,None)
ax.legend(loc="lower left")
ax.text(0.9, 0.89,"(b)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8, transform=ax.transAxes)
fig.tight_layout()
plt.show() 


# In[ ]:





# In[ ]:




