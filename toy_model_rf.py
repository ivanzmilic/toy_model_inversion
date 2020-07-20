import matplotlib.pyplot as plt 
import numpy as np 

def ez_formal_solution(S,tau,eta,profile):
	contrib = np.zeros([len(S),len(profile)]) # invidiual contributions
	w = np.gradient(tau)
	for i in range(0,len(S)): #for each depth:
		contrib[i,:] = S[i] * np.exp(-tau[i]*(1.0+eta*profile)) * w[i] * (1.0+eta*profile)
	emergent_intensity = np.sum(contrib,axis=0)
	return emergent_intensity

def ez_contrib(tau,eta,profile):
	contrib = np.zeros([len(tau),len(profile)]) # invidiual contributions
	w = np.gradient(tau)
	for i in range(0,len(tau)): #for each depth:
		contrib[i,:] = np.exp(-tau[i]*(1.0+eta*profile)) * w[i] * (1.0+eta*profile)
	return contrib

#define the variables that stay the same 
#taugrid:
ND = 81
logtau = np.linspace(-4,1,ND)
tau = 10.**logtau
#line strength:
eta = 50
# wavelength grid, profile
x = np.linspace(-5,5,201) #in reduced wavelength units
profile = 1./np.sqrt(np.pi) * np.exp(-x**2.0)

#now define the source function:
S = np.zeros(ND)
# linearly decreasing
S = 6000.0 + (logtau-1.0) * 1000.0
S /= max(S) #to make things eaier

spectra = ez_formal_solution(S,tau,eta,profile)

plt.clf()
plt.cla()
plt.subplot(211)
plt.plot(logtau,S)
plt.xlabel('$\\log \,\\tau$')
plt.ylabel('$S$')
plt.subplot(212)
plt.plot(x*0.05+6301.5,spectra)
plt.xlabel('$\lambda\,[\,\AA]$')
plt.ylabel('Intensity')
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
plt.tight_layout()
plt.savefig('simple_spectrum_formation.png',fmt='png',bbox_inches='tight')

#now let's see if we can solve the linear system and with what precision
# since contributions are linear in S, basically it is the RF itself, we transpose because we want 
# it in lambda, depth order
rf = ez_contrib(tau,eta,profile).T

# this also means we should be able to infer run of S with the depth from rf and observed spectra directly
# however, the observable has 201 points, we are inferring 51 points in depth. means we need to solve an
# overdetermined linear system. for that we can use the pseudo inverse.
# this method from scipy linalg uses 

plt.clf()
plt.cla()
plt.plot(logtau,S,label='Original')

Nexp  = 30

accuracy = np.zeros(Nexp)
modulus = np.zeros(Nexp)

#criteria = 1E-1

#for i in range(0,Nexp):

rf_pseudo_inv = np.linalg.pinv(rf)
#criteria /= np.sqrt(10.0)

S_solved = np.dot(rf_pseudo_inv,spectra)
#accuracy[i] = np.linalg.norm(spectra - np.dot(rf,S_solved))
#modulus[i] = np.linalg.norm(S_solved-S)

#we now have a new solution. if you started this in ipython you can plot and see the differences
# for now let's just print:
#rel_diff = (S_solved - S) / S

#but we can also plot:
plt.plot(logtau,S_solved,label='Inferred')
#plt.legend()
plt.xlabel('$S$')
plt.ylabel('$\\log\,\\tau$')
plt.legend()

plt.savefig('simple_inversion.png',fmt='png')

