
'''
	Gillespie algorithm (Stochastic simulation algorithm)

	Next-reaction method : J. Phys. Chem. A 104, 1876 (2000)
	
	0. initialize dependance graph G, propensity function a, putative time tau, indexed priority queue P
	1. Choose the reaction having the smallest putative time and corresponding time
	2. Update number of molecules and t=tau
	3. Record trajectory 
	4. Update P and return to 1

	Update of P
	3-1. Update a on G
	3-2. For the fired reaction, sample new random number based on new a so taui=new value+t
	3-2. Otherwise, taui= a_old/a_new(taui-t)+t
	3-3. Update values in P
	
'''
import numpy as np
from copy import deepcopy

class indexed_priority_queue:
	'''
		indexed priority queue
		ref: WilliamFiset Youtube

		i		
		values[i]: value of key
		pm[i]	: position map. heap position of key i
		im[i]	: inverse map. key of heap position i
	'''
	def __init__(self,values=[],keys=[],size=0):
		self.size=size
		self.im=[]
		self.pm=[]
		if len(keys)!=0 and len(values)!=0:
			self.build(keys,values)
	
	def insert(self,ki,value):
		#Insert value into key index (not in previous heap)
		self.values[ki]=value
		self.pm[ki]=self.size
		self.im[self.size]=ki
		self.swim(self.size)
		self.size=self.size+1
		return
	
	def add(self,value):
		#Add new value at the end of heap
		self.insert(self.size+1,value)
		return

	def swap(self,i,j):
		self.pm[self.im[i]]=j	
		self.pm[self.im[j]]=i	
		tmp=self.im[i]
		self.im[i]=self.im[j]
		self.im[j]=tmp
		return

	def parent(self,i):
		return int((i-1)/2) 
	def left(self,i):
		return int(2*i+1)
	def right(self,i):
		return int(2*i+2)

	def less(self,i,j):
		return self.values[self.im[i]]<self.values[self.im[j]]

	def swim(self,i):
		n=i
		p=self.parent(n)
		while n>0 and self.less(n,p):
			self.swap(n,p)
			n=p
			p=self.parent(n)
		return

	def sink(self,i):
		n=i
		while True:
			l=self.left(n)
			r=self.right(n)
			s=l

			if r<self.size and self.less(r,l):
				s=r
			if l>=self.size or self.less(n,s):
				break
	
			self.swap(s,n)
			n=s
		return

	def build(self,keys,values):	
		self.keys=deepcopy(keys)
		self.values=deepcopy(values)
		self.pm=np.arange(self.size).astype(int)
		self.im=np.arange(self.size).astype(int)
		for i in np.arange(0,self.size).astype(int)[::-1]:
			self.sink(i)
			self.swim(i)
		return
	
	def update(self,key,value):
		self.values[key]=value	
		n=self.pm[key]
		self.sink(n)
		self.swim(n)
		return

	def print(self):
		print("values ",self.values)
		print("pm ",self.pm)
		print("im ",self.im)
		return

import itertools	

class next_reaction:
	def __init__(self,model):	
		#The model should have 'proFunc','rxtOrder' and 'changeVec'
		self.model=model
		self.generate_depG()
		self.ipq=indexed_priority_queue(size=len(self.model.proFunc))
			
	def generate_depG(self):
		lf=len(self.model.proFunc)
		self.depGraph=np.zeros((lf,lf)).astype(bool)
		for i,j in itertools.product(range(lf),range(lf)):
				self.depGraph[i,j]=np.any(np.logical_and(testmodel.changeVec[i]!=0 ,testmodel.rxnOrder[j]!=0))
		return
	
	def build_IPQ(self,X):
		
		#Calculate propensity function and corresponding putative time
		nrxn=len(self.model.proFunc)
		self.a=np.zeros(nrxn)
		taus=np.zeros(nrxn)
		for i in range(nrxn):
			self.a[i]=self.model.proFunc[i](X)
			taus[i]=np.random.exponential(1./(self.a[i]+1e-12))

		#Insert data to ipq and bulid
		self.ipq.build(np.arange(nrxn).astype(int),taus)

		return

	def step(self,X):
		#Choose fired reaction
		rxn=self.ipq.im[0]
		tau=self.ipq.values[rxn]
		
		return tau, rxn
	
	def run(self,Xini,tini,tmax,maxstep=None):	
		if len(Xini) != np.shape(self.model.changeVec)[-1]:
			print("State and change vectors has different dimensions!")
			return tini,Xini
		T=np.array([tini])
		X=np.array(np.array([Xini]).T)
		
		#Step 0. Initialization
		x=Xini
		self.build_IPQ(Xini)
		t=tini
		step=0
		goahead=True
		while t<tmax:
			#Step 1. Choose fired reaction and firing time
			tau,ind=self.step(x)
		
			#Step 2. Update
			t=tau
			x=x+self.model.changeVec[ind]		
	
			#Step 3. Record trajectory
			T=np.append(T,t)
			X=np.hstack((X,np.array([x]).T))
			
			if self.model.stop(x):
				break

			if maxstep != None and step+1>=maxstep:
				break
			step=step+1	
			
			#Step 4. Update IPQ	
			update_cand=np.argwhere(self.depGraph[ind])[:,0]
			for i in update_cand:
				anew=self.model.proFunc[i](x)
				if ind != i:
					taunew=(self.a[i]/anew)*(self.ipq.values[i]-t)+t
				else:
					taunew=np.random.exponential(scale=1./(anew+1e-12))+t
				self.a[i]=anew
				self.ipq.update(i,taunew)
			
		#After finishing generation, print data
		return T,X
				
if __name__=="__main__":
	print("Test code for next_reaction.py")		
	from model import model
	'''	
	proFunc=[
	lambda x: x[0]*x[1],
	lambda x: x[1]*x[2],
	lambda x: x[3]*x[4],
	lambda x: x[5],
	lambda x: x[4]*x[6]
	]
	changeVec=np.array([[-1,-1,1,0,0,0,0],
						[0,-1,-1,1,0,0,0],
						[0,0,0,-1,0,1,0],
						[0,0,0,1,0,-1,1],
						[1,0,0,0,-1,0,-1]]).astype(int)
	rxnOrder=np.array([[1,1,0,0,0,0,0],
						[0,1,1,0,0,0,0],
						[0,0,0,1,1,0,0],
						[0,0,0,0,0,1,0],
						[0,0,0,0,1,0,1]]).astype(int)
	
	testmodel=model(proFunc=proFunc,changeVec=changeVec,rxnOrder=rxnOrder)

	solver=next_reaction(model=testmodel)	

	Xini=[1,2,3,4,3,2,1]


	solver.build_IPQ(Xini)	
	solver.ipq.print()
	'''
	'''	
	from direct import direct
	proFunc=[
		lambda x: x[0],
		lambda x: x[1]
	]
	changeVec=np.array([[-1,1,0],[0,-1,1]]).astype(int)
	rxnOrder=np.array([[1,0,0],[0,1,0]]).astype(int)
	Xini=np.array([10000,1,0])
	
	from model import model
	testmodel=model(proFunc=proFunc,rxnOrder=rxnOrder,changeVec=changeVec)
	solver1=next_reaction(model=testmodel)
	solver2=direct(model=testmodel)
	X1s1=[]
	X2s1=[]
	X3s1=[]
	X1s2=[]
	X2s2=[]
	X3s2=[]
	
	import time
	import matplotlib.pyplot as plt
	t1=time.time()
	for i in range(10**2):
		T,X=solver2.run(Xini,0,2.0)
		if i<100:
			plt.plot(T,X[1],lw=1,c='b')
		#plt.plot(T,oc[1],ls='--',lw=1,c='orange')
		X1s2.append(X[0,-1])
		X2s2.append(X[1,-1])
		X3s2.append(X[2,-1])
	t2=time.time()
	for i in range(10**2):
		T,X=solver1.run(Xini,0,2.0)
		if i<100:
			plt.plot(T,X[1],ls='--',lw=1,c='r')#,marker='o',ms=2)
		#plt.plot(T,oc[1],lw=1,c='b')
		X1s1.append(X[0,-1])
		X2s1.append(X[1,-1])
		X3s1.append(X[2,-1])
	t3=time.time()

	plt.ylabel("X[1]")	
	plt.show()
	
	#x=np.arange(25,70)
	plt.hist(X1s1,bins=40,label=r'$Nxt-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.hist(X1s2,bins=40,label=r'$Dir-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.legend(frameon=False)
	plt.show()
	plt.hist(X2s1,bins=40,label=r'$Nxt-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.hist(X2s2,bins=40,label=r'$Dir-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.legend(frameon=False)
	plt.show()
	plt.hist(X3s1,bins=40,label=r'$Nxt-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.hist(X3s2,bins=40,label=r'$Dir-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.legend(frameon=False)
	'''

	#Gamma distribution example
	from direct import direct
	proFunc=[
		lambda x: 0.001*x[0]*x[1],
		lambda x: x[2],
		lambda x: x[3],
		lambda x: x[4]
	]
	changeVec=np.array([[-1,-1,1,0,0,0,],
						[0,0,-1,1,0,0],
						[0,0,0,-1,1,0],
						[1,0,0,0,-1,1]]).astype(int)
	rxnOrder=np.array([[1,1,0,0,0,0],
						[0,0,1,0,0,0],
						[0,0,0,1,0,0],
						[0,0,0,0,1,0]]).astype(int)
	Xini=np.array([1000,1000,0,0,0,0])
	
	from model import model
	testmodel=model(proFunc=proFunc,rxnOrder=rxnOrder,changeVec=changeVec)
	solver1=next_reaction(model=testmodel)
	print(solver1.depGraph)
	solver2=direct(model=testmodel)
	Xs2=np.array([Xini])
	Xs1=np.array([Xini])
		
	import time
	import matplotlib.pyplot as plt
	t1=time.time()
	for i in range(10):
		T,X=solver2.run(Xini,0,2.0)
		if i<100:
			plt.plot(T,X[4],lw=1,c='b')
		#plt.plot(T,oc[1],ls='--',lw=1,c='orange')
		Xs2=np.vstack((Xs2,X[:,-1]))
	t2=time.time()
	for i in range(10):
		T,X=solver1.run(Xini,0,2.0)
		if i<100:
			plt.plot(T,X[4],ls='--',lw=1,c='r')#,marker='o',ms=2)
		#plt.plot(T,oc[1],lw=1,c='b')
		Xs1=np.vstack((Xs1,X[:,-1]))
	t3=time.time()

	plt.ylabel("X[2]")	
	plt.show()
	'''	
	#x=np.arange(25,70)
	plt.hist(Xs1[0],bins=40,label=r'$Nxt-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.hist(Xs2[0],bins=40,label=r'$Dir-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.legend(frameon=False)
	plt.show()
	plt.hist(X2s1,bins=40,label=r'$Nxt-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.hist(X2s2,bins=40,label=r'$Dir-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.legend(frameon=False)
	plt.show()
	plt.hist(X3s1,bins=40,label=r'$Nxt-%fs$'%(t3-t2),histtype=u'step',density=True)
	plt.hist(X3s2,bins=40,label=r'$Dir-%fs$'%(t2-t1),histtype=u'step',density=True)
	plt.legend(frameon=False)

	plt.show()
	'''
