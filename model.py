
'''
	Gillespie algorithm (Stochastic simulation algorithm)

	model:
		Model template to use alorithms.
			

'''
import numpy as np

#template of reaction model for stochastic simulation
class model:
	def __init__(self,proFunc=[],changeVec=[],rxnOrder=[]):
		self.proFunc=proFunc	#List of propensity functions
		self.changeVec=changeVec	#List of change vectors
		self.rxnOrder=rxnOrder	#List of reaction order for each reactant
		self.rxnOrder_t=np.sum(rxnOrder,axis=1) #List of total reaction order
		self.rs=np.argwhere(np.max(rxnOrder,axis=0)!=0)[:,0]

	def replace(self,proFunc,changeVec,rxnOrder):
		self.proFunc=proFunc	#List of propensity functions
		self.changeVec=changeVec	#List of change vectors
		self.rxnOrder=rxnOrder	#List of reaction order for each reactant
		self.rxnOrder_t=np.sum(rxnOrder,axis=1) #List of total reaction order
		self.rs=np.argwhere(np.max(rxnOrder,axis=0)!=0)[:,0]
	
	def proFunc_r(self,X):
		a=np.zeros(len(self.changeVec))
		for i in range(nrxn):
			a[i]=self.model.proFunc[i](X)
		return a
	
	def stop(self,x):
		if np.all(x[self.rs] == 0) or np.any(x<0):
			return True
		else:
			return False	
