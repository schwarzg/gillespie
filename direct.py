import numpy as np

#Direct method
class direct:
	'''
	
		Direct method
		0. Initialize t=t0, x=x0
		1. Calculate propensity function 
		2. Sampling tau and reaction index, calculate change of reactant delta x
		3. Update t=t+tau, x=x+\delta x
		4. Record change and return to step 1
		
	'''
	def __init__(self,model):
		#The model should have 'proFunc' and 'changeVec'
		self.model=model
		
	def	step(self,X):
		#X should have same dimension with model.changeVec
		
		#Step 1. Calculate propensity function
		nrxn=len(self.model.proFunc)
		a=np.zeros(nrxn)
		for i in range(nrxn):
			a[i]=self.model.proFunc[i](X)
		a0=np.sum(a)

		#Step 2. Sampling tau and reaction index
		r1=np.random.uniform(0,1)
		r2=np.random.uniform(0,1)
		tau=-np.log(r1)/(a0+1e-12)
		ind=0
		while np.sum(a[:ind+1])<r2*a0:
			ind=ind+1
	
		return tau,ind
	
	def run(self,Xini,tini,tmax,maxstep=None):
		if len(Xini) != np.shape(self.model.changeVec)[-1]:
			print("State and change vectors has different dimensions!")
			return tini,Xini
		T=np.array([tini])
		X=np.array(np.array([Xini]).T)
		
		#Step 0. Initialization
		x=Xini
		t=tini
		step=0
		goahead=True
		while t<tmax:
			#Step 1,2
			tau,ind=self.step(x)

			#Step 3. Update
			t=t+tau
			x=x+self.model.changeVec[ind]		

	
			#Step 4. Record trajectory
			T=np.append(T,t)
			X=np.hstack((X,np.array([x]).T))
			
			if self.model.stop(x):
				break

			if maxstep != None and step+1>=maxstep:
				break
			step=step+1	

		#After finishing generation, print data
		return T,X

#example - execute if run this file as a main
if __name__=="__main__":
	proFunc=[
		lambda x: x[0],
		lambda x: 1.05*x[1],
		lambda x: 1e-2*x[0]
	]
	
	changeVec=np.array([[1,0],[0,1],[-1,1]]).astype(int)
	rxnOrder=np.array([[1,0],[0,1],[0,1]]).astype(int)
	Xini=np.array([100,0])
	
	from model import model
	test=model(proFunc=proFunc,rxnOrder=rxnOrder,changeVec=changeVec)
	solver=direct(model=test)

	import time
	tik=time.time()
	T,X=solver.run(Xini,0,3)
	print([T,X])
	print(time.time()-tik,"sec")
