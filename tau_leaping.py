import numpy as np
from direct import direct

#tau-leaping method
class tau_leaping:
	'''

		Tau-leaping : J. Chem. Phys. 124, 044109 (2006)
	
		0. initialize X(t0)
		1. Calculate propensity function 
		2. Select time step candidate tau'
		3. if selected tau is too longer than 1/a0, proceed several steps of SSA and return to 1
		4. Sample second candidate tau'' for critical reactions
		5. Determine actual time leap among candidate
		6. Check negativity of species. If taup is to long, check half and return to 3
		7. Update t=t+tau, x=x+\delta x
		8. Record change ahd return to 1
		
		Time step selection in step 2 : J. Chem. Phys. 124, 044109 (2006)
		2-1. count noncritical reaction (ncr) and reactant species (rs)
		2-2. Determine Highest Order of Reaction HOR(i) for each reactant and calculate error parameter epsi=eps/gi
		2-3. Calculate auxlilary quantities
			mui(x)=\sum_j in ncr vij*aj(x)
			sig2i(x)=\sum_j in ncr vij^2aj(x)
		2-4. Calculate taup
			tau=min_i{max{epsi xi/gi,1}/|mui|,max{epsi,gi,1}^2/|sig2i|}
	
	'''
	def __init__(self,model,eps=1e-8,nc=10):
		#The model should have 'proFunc','rxtOrder' and 'changeVec'
		self.model=model
		self.eps=eps
		self.nc=nc
		self.SSA=direct(model=model)

	def set_eps(self,eps):
		self.eps=eps
	def set_nc(self,nc):
		self.nc=nc

	def HOR(self,i):	
		#Highest Order Reaction which species i involve
		return np.max(self.model.rxnOrder_t[self.model.rxnOrder[:,i]!=0])
	
	def	step(self,X):
		#X should have same dimension with model.changeVec
		
		#Step 1. Calculate propensity function
		nrxn=len(self.model.proFunc)
		a=np.zeros(nrxn)
		for i in range(nrxn):
			a[i]=self.model.proFunc[i](X)
		a0=np.sum(a)

		#Step 2. Determine tau
		#2-1. Count ncr and rs
		#may can be improved
		ncr=np.arange(nrxn).astype(int)
		cr=np.array([]).astype(int)
		for j in range(nrxn):
			#calculate maximum number of reaction j
			inds=np.argwhere(self.model.changeVec[j]<0)[:,0]	
			if len(inds)==0:
				continue
			Lj=np.min(int(X[inds]/np.fabs(self.model.changeVec[j,inds])))
			if Lj<self.nc:
				cr=np.append(cr,j)
		ncr=ncr[~np.isin(ncr,cr)]
		ncrs=(np.argwhere(np.sum(self.model.rxnOrder[ncr,:],axis=0)>0)[:,0])

		#2-2. Determine highst order of reaction and error parameter
		g=np.zeros(len(ncrs))
		if len(ncrs)==0:
			taup=np.inf
		else:
			for i in range(len(ncrs)):
				HOR=self.HOR(ncrs[i])
				if HOR==1:
					g[i]=1
				elif HOR==2:
					if np.any(self.model.rxnOrder[:,(np.argwhere(self.model.rxnOrder_t==2)[:,0])]==2):
						g[i]=2.+1./(X[ncrs[i]]-1)
					else:
						g[i]=2.
				elif HOR==3:
					rxnOrder_of_i= self.model.rxnOrder[:,(np.argwhere(self.model.rxnOrder_t==3)[:,0])]
					if np.any(rxnOrder_of_i==2):
						g[i]=1.5*(2.+1./(X[ncrs[i]]-1))
					elif np.any(rxnOrder_of_i==3):
						g[i]=3.+1./(X[ncrs[i]]-1)+2./(X[ncrs[i]]-2)
					else:
						g[i]=3.
				else:
					print("HOR is higher than 3")
					return -1,[-1]

			#2-3. Calculate auxlilary quantities
			mu=np.zeros(len(ncrs))
			sig2=np.zeros(len(ncrs))
			for i in range(len(ncrs)):	#may be problem
				mu[i]=np.dot(self.model.changeVec[ncr,ncrs[i]],a[ncr])
				sig2[i]=np.dot(self.model.changeVec[ncr,ncrs[i]]**2,a[ncr])
	
			#2-4. Calculate tau'
			maxerr=np.maximum(self.eps*X[ncrs]/g,np.ones(len(ncrs)))
			taup=np.min(np.append(maxerr/(1e-12+np.fabs(mu)),maxerr**2/(1e-12+sig2)))

				
		#Step 3. if selected tau is longer enough than 1/a0, choose time step
		while taup>10./a0:
			#Step 4. Sample tau'' for critical reactions
			if len(cr)!=0:
				a0c=np.sum(a[cr])
				taupp=np.random.exponential(scale=1./(a0c+1e-12))		
			else:
				taupp=np.inf
	
			#Step 5. Determine actual time leap
			nrxn=len(self.model.proFunc)		
			numrxn=np.zeros(nrxn)
			if taup<taupp:
				tau=taup
				for j in ncr:
					numrxn[j]=np.random.poisson(lam=a[j]*tau)
			else:
				tau=taupp
				for j in ncr:
					numrxn[j]=np.random.poisson(lam=a[j]*tau)
				jc=np.random.choice(cr,p=a[cr]/a0c)
				numrxn[jc]=1
		
			#Step 6. Check negativity. if taup is to long, check halp
			xnext=X+np.dot(numrxn,self.model.changeVec)
			if np.any(xnext<0):
				taup=taup/2	
				continue
				
			return tau,numrxn
	
		#if taup is to short, return SSA signal numrxn=[-1]
		return 1./a0,[-1]
	
	def run(self,Xini,tini,tmax):
		T=np.array([tini])
		X=np.array(np.array([Xini]).T)
		
		#Step 0. Initialization
		x=Xini
		t=tini
		nssa=0
		while t<tmax :
			#Step 1,2,3,4,5,6
			tau,numrxn=self.step(x)

			if numrxn[-1] != -1 and t+tau>tmax:
				#Step 7. if no SSA signal, update
				t=t+tau
				x=x+np.dot(numrxn,self.model.changeVec)	
			
				#Step 8. Record trajectory
				T=np.append(T,t)
				X=np.hstack((X,np.array([x]).T))
			else:				
				#Step 7'. if SSA signal is returned, do SSA about 100 times until t + 100/a0
				
				###SSA
				tstep=t+1000*tau if t+1000*tau<tmax else tmax
				SSAT,SSAX=self.SSA.run(x,t,tstep,maxstep=100)
				

				#Step 8. Record trajectory
				T=np.append(T,SSAT[1:])
				X=np.hstack((X,SSAX[:,1:]))

				t=T[-1]
				x=X[:,-1]
		
				#nssa=nssa+len(SSAT)

			if self.model.stop(X[:,-1]):
				break
	
		#After finishing generation, print data
		return T,X

#test data
if __name__=="__main__":
	'''	
	proFunc=[
		lambda x: x[0],
		lambda x: 0.002/2*x[0]*(x[0]-1),
		lambda x: 0.5*x[1],
		lambda x: 0.04*x[1],
	]
	changeVec=np.array([[-1,0,0],[-2,1,0],[2,-1,0],[0,-1,1]]).astype(int)
	rxnOrder=np.array([[1,0,0],[2,0,0],[0,1,0],[0,0,1]]).astype(int)
	Xini=np.array([10000,0,0])
	'''
	proFunc=[
		lambda x: x[0],
		lambda x: x[1]
	]
	changeVec=np.array([[-1,1,0],[0,-1,1]]).astype(int)
	rxnOrder=np.array([[1,0,0],[0,1,0]]).astype(int)
	Xini=np.array([10000,1,0])
	
	from model import model
	testmodel=model(proFunc=proFunc,rxnOrder=rxnOrder,changeVec=changeVec)
	solver1=tau_leaping(model=testmodel,eps=0.03)
	solver2=direct(model=testmodel)
	X1s1=[]
	X2s1=[]
	X3s1=[]
	X1s2=[]
	X2s2=[]
	X3s2=[]
	
	import time
	import matplotlib.pyplot as plt
	'''
	for i in range(100):
		T,X=solver1.step(Xini)
		print('First step,',T,X)
	
	'''
	t3=time.time()
	for i in range(10**3):
		T,X=solver2.run(Xini,0,0.2)
		if i<100:
			plt.plot(T,X[1],lw=1,c='b')
		#plt.plot(T,oc[1],ls='--',lw=1,c='orange')
		X1s2.append(X[0,-1])
		X2s2.append(X[1,-1])
		X3s2.append(X[2,-1])
	t1=time.time()
	for i in range(10**3):
		T,X=solver1.run(Xini,0,0.2)
		if i<100:
			plt.plot(T,X[1],ls='--',lw=0,marker='o',ms=2)
		#plt.plot(T,oc[1],lw=1,c='b')
		X1s1.append(X[0,-1])
		X2s1.append(X[1,-1])
		X3s1.append(X[2,-1])
	t2=time.time()
	
	plt.show()
	
	#x=np.arange(25,70)
	plt.hist(X1s1,bins=40,label=r'$Tau-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.hist(X1s2,bins=40,label=r'$Dir-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.legend(frameon=False)
	plt.show()
	plt.hist(X2s1,bins=40,label=r'$Tau-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.hist(X2s2,bins=40,label=r'$Dir-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.legend(frameon=False)
	plt.show()
	plt.hist(X3s1,bins=40,label=r'$Tau-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.hist(X3s2,bins=40,label=r'$Dir-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.legend(frameon=False)
	plt.show()
