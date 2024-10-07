#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("../src/")
# from mfmodeling import NARGP, SingleGP
from NARGP_kern_RBFandExp import NARGP


# ### Read dataset

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df_Nx168 = pd.read_csv("./data/output_ITGae_s_scan_Nx168/time_averaged_qes.csv")
df_Nx336 = pd.read_csv("./data/output_ITGae_s_scan_Nx336/time_averaged_qes.csv")
print(df_Nx168.keys())
print("Low-res. data:", df_Nx168.shape)
print("High-res. data:", df_Nx336.shape)

# Pick up samples
wk1 = df_Nx336.sample(8,random_state=2323)
new_df_Nx336 = wk1

# Training data is only high resolution data
df_train_low = df_Nx168.sort_values('s_hat')
df_train_high = new_df_Nx336.sort_values('s_hat')
df_test_high = df_Nx336.drop(new_df_Nx336.index).sort_values('s_hat')
print("Low-fid. data (train):", df_train_low.shape)
print("High-fid. data (train):", df_train_high.shape)
print("High-fid. data (test):", df_test_high.shape)

fig=plt.figure(figsize=(6,4))
ax = fig.add_subplot()
ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label="Low-res., Nx=168")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label="High-res., Nx=336")
# ax.scatter(df_test_high["s_hat"],df_test_high["averaged Q"],marker="o",facecolor="None",edgecolor="C02",label="High-res., Nx=336")
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$")
ax.set_xlim(-1,1.5)
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
plt.show()


# ### Single GP using only high-res. data

# In[ ]:


df_train_1 = df_train_high.sort_values('s_hat')
data_list_1 = [[np.array(df_train_1)[:,6:7],np.array(df_train_1["averaged Q"])[:,np.newaxis]]]
print(data_list_1[0][0].shape, data_list_1[0][1].shape)
model_singlegp_onlyhigh = NARGP(data_list = data_list_1)             
model_singlegp_onlyhigh.optimize(optimize_restarts=30, max_iters=400)

mean_train_1, var_train_1 = model_singlegp_onlyhigh.predict(np.array(df_train_low)[:,6:7])

fig=plt.figure(figsize=(6,4))
ax = fig.add_subplot()
# ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label="Low-res., Nx=168")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label="High-res., Nx=336")
ax.scatter(df_train_1["s_hat"],df_train_1["averaged Q"],marker="s",facecolor="None",edgecolor="C01",label="Train (only High-res.)")
ax.plot(df_train_low["s_hat"],mean_train_1,".-",color="C03",label="Prediction")
ax.fill_between(df_train_low["s_hat"], (mean_train_1-np.sqrt(var_train_1)).ravel(), (mean_train_1+np.sqrt(var_train_1)).ravel(), alpha=0.1, color='C03')
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$")
ax.set_xlim(-1,1.5)
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
plt.show()


# ### Single GP using mixed data
# When high-res. data is available, replace low-res. data by high-res. one.

# In[ ]:


# Training data is mixed data, replacing low-res. data by high-res. data when available.
wkdf = pd.concat([df_train_low[df_train_low["Directory Name"]==x] for x in df_train_high["Directory Name"]])
wkdf2 = df_train_low.drop(wkdf.index)
df_train_mixed = pd.concat([wkdf2,df_train_high])

df_train_2 = df_train_mixed.sort_values('s_hat')
data_list_2 = [[np.array(df_train_2)[:,6:7],np.array(df_train_2["averaged Q"])[:,np.newaxis]]]
print(data_list_2[0][0].shape, data_list_2[0][1].shape)
model_singlegp_mixed = NARGP(data_list = data_list_2)             
model_singlegp_mixed.optimize(optimize_restarts=30, max_iters=400)

mean_train_2, var_train_2 = model_singlegp_mixed.predict(np.array(df_train_low)[:,6:7])

fig=plt.figure(figsize=(6,4))
ax = fig.add_subplot()
ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label="Low-res., Nx=168")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label="High-res., Nx=336")
ax.scatter(df_train_2["s_hat"],df_train_2["averaged Q"],marker="s",facecolor="None",edgecolor="C01",label="Train (mixed data)")
ax.plot(df_train_low["s_hat"],mean_train_2,".-",color="C03",label="Prediction")
ax.fill_between(df_train_low["s_hat"], (mean_train_2-np.sqrt(var_train_2)).ravel(), (mean_train_2+np.sqrt(var_train_2)).ravel(), alpha=0.1, color='C03')
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$")
ax.set_xlim(-1,1.5)
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
plt.show()


# ### NARGP using both low-res. and high-res. data

# In[ ]:


data_list = [[np.array(df_train_low)[:,6:7],np.array(df_train_low["averaged Q"])[:,np.newaxis]],
             [np.array(df_train_high)[:,6:7],np.array(df_train_high["averaged Q"])[:,np.newaxis]]]
print(data_list[0][0].shape, data_list[0][1].shape)
print(data_list[1][0].shape, data_list[1][1].shape)
model_nargp = NARGP(data_list = data_list)             
model_nargp.optimize(optimize_restarts=30, max_iters=400)

mean_train, var_train = model_nargp.predict(np.array(df_train_low)[:,6:7])

fig=plt.figure(figsize=(6,4))
ax = fig.add_subplot()
ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label="Low-res., Nx=168")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label="High-res., Nx=336")
ax.scatter(data_list[0][0],data_list[0][1],marker="s",facecolor="None",edgecolor="C01",label="Train (all data)")
ax.scatter(data_list[1][0],data_list[1][1],marker="s",facecolor="None",edgecolor="C01",label="")
ax.plot(df_train_low["s_hat"],mean_train,".-",color="C03",label="Prediction")
ax.fill_between(df_train_low["s_hat"], (mean_train-np.sqrt(var_train)).ravel(), (mean_train+np.sqrt(var_train)).ravel(), alpha=0.1, color='C03')
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$")
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
plt.show()


# ### Plot Figure 2

# In[ ]:


plt.style.use('../nature_style.txt')

fig=plt.figure(figsize=(5.9,4),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
ax=fig.add_subplot(2,2,1)
ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label=r"Low-res. $Q_\mathrm{L}$")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label=r"High-res. $Q_\mathrm{H}$")
# ax.scatter(df_test_high["s_hat"],df_test_high["averaged Q"],marker="o",facecolor="None",edgecolor="C02",label=r"High-res. $Q_\mathrm{H}$")
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$ [$Q_\mathrm{gB}$]")
ax.set_xlim(-1,1.5)
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
ax.text(-0.92,89,"(a)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

ax=fig.add_subplot(2,2,2)
# ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label=r"Low-res. $Q_\mathrm{L}$")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label=r"High-res. $Q_\mathrm{H}$")
ax.scatter(df_train_1["s_hat"],df_train_1["averaged Q"],marker="s",facecolor="None",edgecolor="C01",label="Training (High-res.)")
ax.plot(df_train_low["s_hat"],mean_train_1,".-",color="C03",label="Single GP pred.")
ax.fill_between(df_train_low["s_hat"], (mean_train_1-np.sqrt(var_train_1)).ravel(), (mean_train_1+np.sqrt(var_train_1)).ravel(), alpha=0.1, color='C03')
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$ [$Q_\mathrm{gB}$]")
ax.set_xlim(-1,1.5)
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
ax.text(-0.92,89,"(b)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

ax=fig.add_subplot(2,2,3)
ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label=r"Low-res. $Q_\mathrm{L}$")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label=r"High-res. $Q_\mathrm{H}$")
ax.scatter(df_train_2["s_hat"],df_train_2["averaged Q"],marker="s",facecolor="None",edgecolor="C01",label="Training (mixed)")
ax.plot(df_train_low["s_hat"],mean_train_2,".-",color="C03",label="Single GP pred.")
ax.fill_between(df_train_low["s_hat"], (mean_train_2-np.sqrt(var_train_2)).ravel(), (mean_train_2+np.sqrt(var_train_2)).ravel(), alpha=0.1, color='C03')
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$ [$Q_\mathrm{gB}$]")
ax.set_xlim(-1,1.5)
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
ax.text(-0.92,89,"(c)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

ax=fig.add_subplot(2,2,4)
ax.plot(df_train_low["s_hat"],df_train_low["averaged Q"],"+",color="C00",label=r"Low-res. $Q_\mathrm{L}$")
ax.plot(df_train_high["s_hat"],df_train_high["averaged Q"],"x",color="C02",label=r"High-res. $Q_\mathrm{H}$")
ax.scatter(data_list[0][0],data_list[0][1],marker="s",facecolor="None",edgecolor="C01",label="Training (all)")
ax.scatter(data_list[1][0],data_list[1][1],marker="s",facecolor="None",edgecolor="C01",label="")
ax.plot(df_train_low["s_hat"],mean_train,".-",color="C03",label="NARGP pred.")
ax.fill_between(df_train_low["s_hat"], (mean_train-np.sqrt(var_train)).ravel(), (mean_train+np.sqrt(var_train)).ravel(), alpha=0.1, color='C03')
ax.set_xlabel(r"Magnetic shear $\hat{s}$")
ax.set_ylabel(r"Heat flux $Q$ [$Q_\mathrm{gB}$]")
ax.set_xlim(-1,1.5)
ax.set_ylim(0,100)
ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()
ax.text(-0.92,89,"(d)",color="k", fontfamily="sans-serif", fontweight="bold", fontsize=8)

fig.tight_layout()
plt.savefig("fig2.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# ### Plot Figure 3(a)

# In[ ]:


m1 = model_nargp.model_list[1]
s_hat = np.linspace(-1.0,1.5,100)
Q_low = np.linspace(5,95,100)
x2,y2=np.meshgrid(s_hat,Q_low)
meshdata = np.hstack((x2.reshape([10000,1]), y2.reshape([10000,1])))
# print(x2.shape,meshdata.shape)
mu, var = m1.predict(meshdata)

wk_Nx168 = pd.concat([df_Nx168[df_Nx168["Directory Name"]==x] for x in df_Nx336["Directory Name"]])
wk_Nx168 = wk_Nx168.sort_values("s_hat")
wk_Nx336 = df_Nx336.sort_values("s_hat")
wk_new_Nx168 = pd.concat([df_Nx168[df_Nx168["Directory Name"]==x] for x in new_df_Nx336["Directory Name"]])
wk_new_Nx168 = wk_new_Nx168.sort_values("s_hat")
wk_new_Nx336 = new_df_Nx336.sort_values("s_hat")


plt.style.use('../nature_style.txt')

# fig=plt.figure(figsize=(3.3,3.3),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
fig=plt.figure(figsize=(4,4),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
ax = fig.add_subplot(projection="3d")
ax.view_init(elev=20, azim=-130, roll=0)
# ax.plot(wk_Nx168["s_hat"],wk_Nx168["averaged Q"],wk_Nx336["averaged Q"],"o-")
ax.plot_surface(x2,y2,mu.reshape([100,100]),cmap="viridis",alpha=0.5)
ax.plot(wk_Nx168["s_hat"],wk_Nx168["averaged Q"],wk_Nx336["averaged Q"],".-",color="C0")
ax.plot(wk_new_Nx168["s_hat"],wk_new_Nx168["averaged Q"],wk_new_Nx336["averaged Q"],"s",color="C1")
ax.set_xlabel(r"$\hat s$")
ax.set_ylabel(r"Low-res. $Q_\mathrm{L}$")
ax.set_zlabel(r"High-res. $Q_\mathrm{H}$")
# ax.set_ylim(0,100)
# ax.set_zlim(0,100)
# ax.set_aspect("equal")
# ax.plot([0,100],[0,100],"--",color="k",lw=0.5)
ax.set_box_aspect(None, zoom=0.7)

plt.tight_layout()
plt.savefig("fig3a.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# ### Plot Figure 3(b)

# In[ ]:


# plt.style.use('../nature_style.txt')

# fig=plt.figure(figsize=(3.3,2.2),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
# ax = fig.add_subplot()
# ax.plot(df_Nx168.sort_values("s_hat")["s_hat"],df_Nx168.sort_values("s_hat")["averaged Q"],"+",lw=1,label=r"Low-res. $Q_\mathrm{L}$")
# ax.plot(df_Nx336.sort_values("s_hat")["s_hat"],df_Nx336.sort_values("s_hat")["averaged Q"],"x-",lw=1,color="C02",label=r"High-res. $Q_\mathrm{H}$")
# ax.plot(df_train_low["s_hat"],mean_train,".-",color="C03",label="NARGP pred.")
# ax.fill_between(df_train_low["s_hat"], (mean_train-np.sqrt(var_train)).ravel(), (mean_train+np.sqrt(var_train)).ravel(), alpha=0.1, color='C03')
# ax.set_xlabel(r"Magnetic shear $\hat{s}$")
# ax.set_ylabel(r"Heat flux $Q$ [$Q_\mathrm{gB}$]")
# ax.set_xlim(-1,1.5)
# ax.set_ylim(0,100)
# ax.axvline(0,linestyle="--",color="k",lw=0.5)
# ax.legend()

# plt.tight_layout()
# plt.savefig("fig3b.pdf",dpi=600,bbox_inches="tight")
# plt.show() 


# In[ ]:


plt.style.use('../nature_style.txt')

mean1, _ = model_singlegp_onlyhigh.predict(np.array(df_Nx336)[:,6:7])
mean2, _ = model_singlegp_mixed.predict(np.array(df_Nx336)[:,6:7])
mean3, _ = model_nargp.predict(np.array(df_Nx336)[:,6:7])

fig=plt.figure(figsize=(2.8,2.8),dpi=600) # figsize=(width,height(inch)),dpi(dots per inch)
ax = fig.add_subplot()
# ax.plot(df_Nx168.sort_values("s_hat")["s_hat"],df_Nx168.sort_values("s_hat")["averaged Q"],"+",lw=1,label=r"Low-res. $Q_\mathrm{L}$")
ax.plot(mean1,df_Nx336["averaged Q"],"+",lw=1,label="Single GP (High-res. data)")
ax.plot(mean2,df_Nx336["averaged Q"],"x",lw=1,label="Single GP (mixed data)")
ax.plot(mean3,df_Nx336["averaged Q"],".",lw=1,label="NARGP")
vmin = 0 #df_Nx336["averaged Q"].min()
vmax = 60 #df_Nx336["averaged Q"].max()
ax.plot([vmin,vmax],[vmin,vmax],linestyle="--",color="k",lw=0.5)
ax.set_aspect("equal")
# ax.plot(df_train_low["s_hat"],mean_train,".-",color="C03",label="NARGP pred.")
# ax.fill_between(df_train_low["s_hat"], (mean_train-np.sqrt(var_train)).ravel(), (mean_train+np.sqrt(var_train)).ravel(), alpha=0.1, color='C03')
ax.set_xlabel(r"Prediction $Q$ [$Q_\mathrm{gB}$]")
ax.set_ylabel(r"High-res. $Q_\mathrm{H}$ [$Q_\mathrm{gB}$]")
ax.set_xlim(vmin,vmax)
ax.set_ylim(vmin,vmax)
# ax.axvline(0,linestyle="--",color="k",lw=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("fig3b.pdf",dpi=600,bbox_inches="tight")
plt.show() 


# In[ ]:


ans = df_Nx336["averaged Q"]
R2_1 = 1 - np.sum((ans-mean1.ravel())**2)/np.sum((ans-ans.mean())**2)
R2_2 = 1 - np.sum((ans-mean2.ravel())**2)/np.sum((ans-ans.mean())**2)
R2_3 = 1 - np.sum((ans-mean3.ravel())**2)/np.sum((ans-ans.mean())**2)
print(R2_1, R2_2, R2_3)


# In[ ]:




