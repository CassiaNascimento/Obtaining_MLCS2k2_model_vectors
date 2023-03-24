import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sncosmo
import glob
import scipy
import math
import GPy
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import extinction

# Defining Hsiao Flux

hsiao_template=pd.read_csv("/home/cassia/SNANA/snroot/snsed/Hsiao07.dat",header=None,sep="\s+")
hsiao_template.columns=["phase","lambda","flux"]

hsiao_template=hsiao_template[(hsiao_template["phase"]>-11.) & (hsiao_template["phase"]<51.)]

Hsiao_temp_phase=[]
Hsiao_flux={}
Hsiao_flux_U=[]
Hsiao_flux_B=[]
Hsiao_flux_V=[]
Hsiao_flux_R=[]
Hsiao_flux_I=[]

for d in hsiao_template["phase"].unique():
    spec=sncosmo.Spectrum(wave=hsiao_template[hsiao_template["phase"]==d]["lambda"].values, flux=hsiao_template[hsiao_template["phase"]==d]["flux"].values)
    Hsiao_temp_phase.append(d)
    Hsiao_flux_U.append(spec.bandflux("standard::u",zp=27.5,zpsys="ab"))
    Hsiao_flux_B.append(spec.bandflux("standard::b",zp=27.5,zpsys="ab"))
    Hsiao_flux_V.append(spec.bandflux("standard::v",zp=27.5,zpsys="ab"))
    Hsiao_flux_R.append(spec.bandflux("standard::r",zp=27.5,zpsys="ab"))
    Hsiao_flux_I.append(spec.bandflux("standard::i",zp=27.5,zpsys="ab"))
    
Hsiao_flux["U"]=Hsiao_flux_U
Hsiao_flux["B"]=Hsiao_flux_B
Hsiao_flux["V"]=Hsiao_flux_V
Hsiao_flux["R"]=Hsiao_flux_R
Hsiao_flux["I"]=Hsiao_flux_I

fig,axs= plt.subplots(1,5,figsize=(15,3))
for i,filt in enumerate(["U","B","V","R","I"]):
    
    axs[i].plot(Hsiao_temp_phase,Hsiao_flux[filt])
    axs[i].set_xlabel("Phase (days)")
    axs[i].set_ylabel(r"Template "+filt+" ($ZP_{AB}=27.5$)")
    
plt.tight_layout();
plt.savefig("./outputs/Hsiao_templates.png")
plt.close(fig)

# MW Extinction

lambda_eff={"U":[3600.],"B":[4500.],"V":[5500.],"R":[6600.],"I":[8000.]}
RV=3.1
def mw_ext_cor(mwebv):
    ext={"U":extinction.ccm89(np.array(lambda_eff["U"]),mwebv*RV,RV)[0],
    "B":extinction.ccm89(np.array(lambda_eff["B"]),mwebv*RV,RV)[0],
    "V":extinction.ccm89(np.array(lambda_eff["V"]),mwebv*RV,RV)[0],
    "R":extinction.ccm89(np.array(lambda_eff["R"]),mwebv*RV,RV)[0],
    "I":extinction.ccm89(np.array(lambda_eff["I"]),mwebv*RV,RV)[0]}
    return ext

# Reading LCs and performing GPR

paths=glob.glob("/home/cassia/SNANA/snroot/lcmerge/LOWZ_JRK07/*.DAT")

path="/home/cassia/SNANA/snroot/lcmerge/LOWZ_JRK07/LOWZ_JRK07_"

name=list()
for i in range(len(paths)):
    name.append(paths[i].split('_')[3].split('.')[0])

train=["1990O","1990af","1992P","1992ae","1992al","1992bc","1992bg","1992bl","1992bo","1992bp","1992br","1993H","1993O","1993ag","1994M","1995ac","1996C","1996bl","1997E","1997Y","1997bq","1998V","1998ab","1998bp","1998de","1998es","1999aa","1999ac","1999da","1999gp","2000ca","2000cf","2000dk","2000fa","2001V","2001cz","2002er"]

gnoise={"1990O":0.,"1990af":0.001,"1992P":0.01,"1992ae":0.01,"1992al":0.001,"1992bc":0.001,"1992bg":0.001,"1992bl":0.001,"1992bo":0.001,"1992bp":0.001,"1992br":0.005,"1993H":0.001,
        "1993O":0.001,"1993ag":0.005,"1994M":0.001,"1995ac":0.001,"1996C":0.001,"1996bl":0.001,"1997E":0.001,"1997Y":0.001,"1997bq":0.001,"1998V":0.001,"1998ab":0.001,"1998bp":0.001,"1998de":0.005,
        "1998es":0.005,"1999aa":0.001,"1999ac":0.005,"1999da":0.01,"1999gp":0.005,"2000ca":0.001,"2000cf":0.001,"2000dk":0.005,"2000fa":0.001,"2001V":0.001,"2001cz":0.001,"2002er":0.005}

ignore_filter={"1990O":[],"1990af":[],"1992P":[],"1992ae":[],"1992al":[],"1992bc":[],"1992bg":[],"1992bl":[],"1992bo":[],"1992bp":["V"],"1992br":["B"],"1993H":[],
        "1993O":[],"1993ag":[],"1994M":["I"],"1995ac":[],"1996C":[],"1996bl":[],"1997E":[],"1997Y":[],"1997bq":[],"1998V":[],"1998ab":[],"1998bp":[],"1998de":["U"],
        "1998es":["B"],"1999aa":[],"1999ac":[],"1999da":[],"1999gp":[],"2000ca":[],"2000cf":[],"2000dk":[],"2000fa":[],"2001V":[],"2001cz":["I"],"2002er":[]}

# Incertezas pequenas, dificuldade em minimizar o chi quadrado para encaixar o template Hsiao
# 1992P I band?
# 1995ac B band?
# 1999da V band?

phase=np.arange(-10,51,1)

def gp_w_prior(SN,f,p,mag,magerr):
    kern1=GPy.kern.Matern52(1,variance=0.1, lengthscale=15.0)#+GPy.kern.Fixed(1,np.diag(magerr**2))

    temp=interp1d(Hsiao_temp_phase,Hsiao_flux[f]) # interpolando o template
       
    def chi2(theta):
        a,c =theta
        return np.sum((mag-(a*temp(p)+c))**2/magerr**2)

    res=minimize(chi2,[1,0],method="Nelder-Mead")
    result=[res.x[0],res.x[1]]

    def mean(x):
        return result[0]*temp(x)+result[1]


    mf = GPy.core.Mapping(1,1)
    mf.f = mean
    mf.update_gradients = lambda a,b: None
            
    model=GPy.models.GPRegression(p.reshape(-1, 1),mag.reshape(-1, 1),kernel=kern1,mean_function=mf)
    
    model.Gaussian_noise.variance.fix(gnoise[SN])
    prior1 = GPy.priors.Gaussian(mu=25.,sigma=1.)
    prior2 = GPy.priors.Gaussian(mu=0.1,sigma=0.1)#0.005)
    prior1.domain="positive"
    prior2.domain="positive"

    model.kern.lengthscale.set_prior(prior1,warning=False)
    model.kern.variance.set_prior(prior2,warning=False)

    #model.kern.Mat52.lengthscale.set_prior(prior1,warning=False)
    #model.kern.Mat52.variance.set_prior(prior2,warning=False)
    #model.kern.fixed.variance.fix()
    
    model.optimize()
    #print(model)
    
    kern3 = GPy.kern.Matern52(1,variance= model.param_array[0], lengthscale=model.param_array[1])
    
    y_mean, y_var=model.predict(np.array(phase).reshape(-1, 1),kern=kern3)

    return y_mean, y_var, temp, result

def GPR(SN):
      
    meta, lc_data=sncosmo.read_snana_ascii(path+SN+'.DAT',default_tablename="OBS")
    lc_data=lc_data["OBS"].to_pandas()    
    red=str(meta["REDSHIFT_FINAL"])
    peak_mjd=meta["SEARCH_PEAKMJD"]
    mwebv=meta["MWEBV"]
    ext=mw_ext_cor(mwebv)

    lc_data["(MJD-PEAK)/(1+z)"]=(lc_data["MJD"]-peak_mjd)/(1+float(red))
    lc_data=lc_data[(lc_data["(MJD-PEAK)/(1+z)"]>=-10) & (lc_data["(MJD-PEAK)/(1+z)"]<=50)]

    filters=list()
    filt=lc_data["FLT"].unique()
    for f in ["U","B","V","R","I"]:
        if f in filt:
            if f not in ignore_filter[SN]:
                filters.append(f)

    filters_colors={"U":"C0","B":"C1","V":"C2","R":"C3","I":"C4"}
    
    i=0
    j=0
    
    if len(filters)==1:
        fig, axs = plt.subplots(1, len(filters),figsize=(10,3))
        fig.suptitle("SN "+SN+', z: '+red)

        for f in filters:
            p=lc_data[lc_data["FLT"]==f]["(MJD-PEAK)/(1+z)"].values
            mag=-2.5*np.log10(lc_data[lc_data["FLT"]==f]["FLUXCAL"].values*10**(0.4*ext[f]))+27.5
            magerr=lc_data[lc_data["FLT"]==f]["MAGERR"].values # nao acertei ainda o valor
            
            y_mean, y_var, temp, result = gp_w_prior(SN,f,p,mag,magerr)
            np.savetxt("./outputs/dat/SN"+SN+"_GPR_filter_"+f+".dat",y_mean)

            axs.errorbar(p,mag,yerr=magerr,ls='none',marker='.',markersize=3.,c=filters_colors[f],label="Data")
            axs.plot(phase,y_mean.reshape(-1),label="GPR",c=filters_colors[f])
            axs.fill_between(phase,y_mean.reshape(-1)-2.*np.sqrt(y_var).reshape(-1),y_mean.reshape(-1)+2.*np.sqrt(y_var).reshape(-1),label="95% confidence",color=filters_colors[f],alpha=0.4)
            axs.plot(phase,result[0]*temp(phase)+result[1],label="Hsiao07 "+f+" template",c="C7")
            axs.legend()
            axs.set_xlabel(r"$(t-t_0)/(1+z)$")
            axs.set_ylabel("Magnitude ($ZP_{AB}=27.5$)")
            axs.invert_yaxis()
            i=i+1
        fig.tight_layout()
        plt.savefig("./outputs/"+SN+"_GPR_plot.png")
        plt.close(fig)

    elif 1<len(filters)<3:
        fig, axs = plt.subplots(1, len(filters),figsize=(10,3))
        fig.suptitle("SN "+SN+', z: '+red)

        for f in filters:
            p=lc_data[lc_data["FLT"]==f]["(MJD-PEAK)/(1+z)"].values
            mag=-2.5*np.log10(lc_data[lc_data["FLT"]==f]["FLUXCAL"].values*10**(0.4*ext[f]))+27.5
            magerr=lc_data[lc_data["FLT"]==f]["MAGERR"].values # nao acertei ainda o valor
            
            y_mean, y_var, temp, result = gp_w_prior(SN,f,p,mag,magerr)
            np.savetxt("./outputs/dat/SN"+SN+"_GPR_filter_"+f+".dat",y_mean)

            axs[i].errorbar(p,mag,yerr=magerr,ls='none',marker='.',markersize=3.,c=filters_colors[f],label="Data")
            axs[i].plot(phase,y_mean.reshape(-1),label="GPR",c=filters_colors[f])
            axs[i].fill_between(phase,y_mean.reshape(-1)-2.*np.sqrt(y_var).reshape(-1),y_mean.reshape(-1)+2.*np.sqrt(y_var).reshape(-1),label="95% confidence",color=filters_colors[f],alpha=0.4)
            axs[i].plot(phase,result[0]*temp(phase)+result[1],label="Hsiao07 "+f+" template",c="C7")
            axs[i].legend()
            axs[i].set_xlabel(r"$(t-t_0)/(1+z)$")
            axs[i].set_ylabel("Magnitude ($ZP_{AB}=27.5$)")
            axs[i].invert_yaxis()
            i=i+1
        fig.tight_layout()
        plt.savefig("./outputs/"+SN+"_GPR_plot.png")
        plt.close(fig)

    elif len(filters)<5:
        fig, axs = plt.subplots(2, 2,figsize=(10,6))
        fig.suptitle("SN "+SN+', z: '+red)

        if len(filters)==3:
            axs[1,1].set_axis_off()
        for f in filters:
            p=lc_data[lc_data["FLT"]==f]["(MJD-PEAK)/(1+z)"].values
            mag=-2.5*np.log10(lc_data[lc_data["FLT"]==f]["FLUXCAL"].values*10**(0.4*ext[f]))+27.5
            magerr=lc_data[lc_data["FLT"]==f]["MAGERR"].values
            
            y_mean, y_var, temp, result = gp_w_prior(SN,f,p,mag,magerr)
            np.savetxt("./outputs/dat/SN"+SN+"_GPR_filter_"+f+".dat",y_mean)

            if j==1:
                axs[i, j].errorbar(p,mag,yerr=magerr,ls='none',marker='.',markersize=3.,c=filters_colors[f],label="Data")
                axs[i, j].plot(phase,y_mean.reshape(-1),label="GPR",c=filters_colors[f])
                axs[i, j].fill_between(phase,y_mean.reshape(-1)-2.*np.sqrt(y_var).reshape(-1),y_mean.reshape(-1)+2.*np.sqrt(y_var).reshape(-1),label="95% confidence",color=filters_colors[f],alpha=0.4)
                axs[i, j].plot(phase,result[0]*temp(phase)+result[1],label="Hsiao07 "+f+" template",c="C7")
                axs[i, j].legend()
                axs[i, j].set_xlabel(r"$(t-t_0)/(1+z)$")
                axs[i, j].set_ylabel("Magnitude ($ZP_{AB}=27.5$)")
                axs[i, j].invert_yaxis()
                i=i+1
                j=0
            else: 
                axs[i, j].errorbar(p,mag,yerr=magerr,ls='none',marker='.',markersize=3.,c=filters_colors[f],label="Data")
                axs[i, j].plot(phase,y_mean.reshape(-1),label="GPR",c=filters_colors[f])
                axs[i, j].fill_between(phase,y_mean.reshape(-1)-2.*np.sqrt(y_var).reshape(-1),y_mean.reshape(-1)+2.*np.sqrt(y_var).reshape(-1),label="95% confidence",color=filters_colors[f],alpha=0.4)
                axs[i, j].plot(phase,result[0]*temp(phase)+result[1],label="Hsiao07 "+f+" template",c="C7")
                axs[i, j].legend()
                axs[i, j].set_xlabel(r"$(t-t_0)/(1+z)$")
                axs[i, j].set_ylabel("Magnitude ($ZP_{AB}=27.5$)")
                axs[i, j].invert_yaxis()
                j=j+1

        fig.tight_layout()
        plt.savefig("./outputs/"+SN+"_GPR_plot.png")
        plt.close(fig)

    else:
        fig, axs = plt.subplots(3, 2,figsize=(10,9))
        fig.suptitle("SN "+SN+', z: '+red)
        axs[2,1].set_axis_off()
        
        for f in filters:
            p=lc_data[lc_data["FLT"]==f]["(MJD-PEAK)/(1+z)"].values
            mag=-2.5*np.log10(lc_data[lc_data["FLT"]==f]["FLUXCAL"].values*10**(0.4*ext[f]))+27.5
            magerr=lc_data[lc_data["FLT"]==f]["MAGERR"].values

            y_mean, y_var, temp, result = gp_w_prior(SN,f,p,mag,magerr)
            np.savetxt("./outputs/dat/SN"+SN+"_GPR_filter_"+f+".dat",y_mean)

            if j==1:
                axs[i, j].errorbar(p,mag,yerr=magerr,ls='none',marker='.',markersize=3.,c=filters_colors[f],label="Data")
                axs[i, j].plot(phase,y_mean.reshape(-1),label="GPR",c=filters_colors[f])
                axs[i, j].fill_between(phase,y_mean.reshape(-1)-2.*np.sqrt(y_var).reshape(-1),y_mean.reshape(-1)+2.*np.sqrt(y_var).reshape(-1),label="95% confidence",color=filters_colors[f],alpha=0.4)
                axs[i, j].plot(phase,result[0]*temp(phase)+result[1],label="Hsiao07 "+f+" template",c="C7")
                axs[i, j].legend()
                axs[i, j].set_xlabel(r"$(t-t_0)/(1+z)$")
                axs[i, j].set_ylabel("Magnitude ($ZP_{AB}=27.5$)")
                axs[i, j].invert_yaxis()
                i=i+1
                j=0
            else: 
                axs[i, j].errorbar(p,mag,yerr=magerr,ls='none',marker='.',markersize=3.,c=filters_colors[f],label="Data")
                axs[i, j].plot(phase,y_mean.reshape(-1),label="GPR",c=filters_colors[f])
                axs[i, j].fill_between(phase,y_mean.reshape(-1)-2.*np.sqrt(y_var).reshape(-1),y_mean.reshape(-1)+2.*np.sqrt(y_var).reshape(-1),label="95% confidence",color=filters_colors[f],alpha=0.4)
                axs[i, j].plot(phase,result[0]*temp(phase)+result[1],label="Hsiao07 "+f+" template",c="C7")
                axs[i, j].legend()
                axs[i, j].set_xlabel(r"$(t-t_0)/(1+z)$")
                axs[i, j].set_ylabel("Magnitude ($ZP_{AB}=27.5$)")
                axs[i, j].invert_yaxis()
                j=j+1

        fig.tight_layout()
        plt.savefig("./outputs/"+SN+"_GPR_plot.png")
        plt.close(fig)

for SN in train:
    GPR(SN)
    #plt.show()
 