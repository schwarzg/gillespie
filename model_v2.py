'''
	Gillespie algorithm (Stochastic simulation algorithm)

	model:
		Model template to use alorithms.

	v2
		Now, proFunc is a function that return propensity function values in array			

'''
import numpy as np

#template of reaction model for stochastic simulation
class model:
	def __init__(self,proFunc,changeVec=[],rxnOrder=[]):
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
	
	def stop(self,x):
		if np.all(x[self.rs] == 0) or np.any(x<0):
			return True
		else:
			return False	
