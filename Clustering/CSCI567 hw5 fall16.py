import sys
import csv,random
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvnorm


def check_cluster(c1,c2,k):
	for i in range(0,k):
		for j in c1[i]:
			if j not in c2[i]:
				return True
	return False

def euclidean(p1,p2):
	return np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))

def add_to_class(classes,point,i):

	if i in classes:

		classes[i].append(point)
	else:
		classes[i] = []
		classes[i].append(point)
	return classes

def assign(points_np,center,c_index,K):
	prev_class = {}
	check  = True
	while(check):
		classes = {}
		for i in range(0,len(points_np)):	
			dist = []	
			for cen in range(0,K):
			 	dist.append(euclidean(points_np[i],center[cen]))

			ind  = dist.index(min(dist))
			classes = add_to_class(copy.copy(classes),points_np[i],ind)
		for i in classes:
			classes[i] = np.array(classes[i])
		if not prev_class:
			prev_class = classes
			for k in range(0,K):
				center[k] = [np.mean(classes[k][:,0]),np.mean(classes[k][:,1])]
			continue
		else:
			check = check_cluster(classes,prev_class,K)
			for k in range(0,K):
				center[k] = [np.mean(classes[k][:,0]),np.mean(classes[k][:,1])]
			prev_class = classes
	return center,classes

def assign_cluster(points,k):
	centers = []
	c_index = []
	for i in range(0,k):
		center = random.choice(points)
		c_index.append(points.index(center))
		centers.append(center)
	points_np = np.array(points)
	c,d = assign(points_np,copy.copy(centers),c_index,k)
	return c,d

def to_numpy(data):
	for i in range(0,len(data)):
		data[i] = np.array(data[i])
	return np.array(data)

def get_points(data):
	points = []
	for i in data:
		k = i.split(',')
		for j in range(0,2):
			k[j] = float(k[j])
		points.append((k))
	return points

def kmeans(fread):
	data = file.read(file(fread)).splitlines()
	points = get_points(data)
	K = [2,3,5]
	color = ['red','blue','green','yellow','brown']
	for i in range(0,len(K)):
		plt.figure(i)
		center,classes=assign_cluster(copy.copy(points),K[i])
		for j in  range(K[i]):
			x,y = classes[j][:,0],classes[j][:,1]
			plt.scatter(x,y,color=color[j])
		plt.show()

def calc_gmm(X,mu,sigma,pi):
	pass

def calc_cov(p,pi,mu,K):
	total = 0.0
	for i in range(len(p)):
		diff = np.subtract(p[i],mu[K])
		total+= diff[0]*diff[1]
	return total/(pi[K]*600.0)


# pi -> List of pi for all clusters , pdf -> probability of 600 points belonging to each cluster

def E_Step(pi,pdf):
	ric  = np.zeros((len(pdf[0]),len(pi)))
	for i in range(0,len(pdf[0])):
		for j in range(0,len(pi)):
			nr =  pi[j]* pdf[j][i]
			dr = 0.0
			for k in range(0,len(pi)):
				dr+= pi[k]* pdf[k][i]
			ric[i][j] = float(nr/dr)
	return ric

def M_Step(p,ric):
	mc = np.zeros(ric.shape[1])
	for i in range(ric.shape[1]):
		mc[i] = sum(ric[:,i])
	pi = []
	for k in mc:
		pi.append(k/sum(mc))
	muc =[]
	for i in range(3):
		nr =np.zeros(2)

		for j in range(len(p)):
			nr += p[j]* ric[j][i]
			
		muc.append(nr/mc[i])
	sigma = []
	for i in range(3):
		nr = np.zeros((2,2))
		for j in range(len(p)):
			nr += ric[j][i] * np.outer(np.transpose((p[j] - muc[i])),(p[j] - muc[i]))
		sigma.append((nr)/float(mc[i]))
	return pi,muc,sigma

def logit(pi,p,mu,si,old):
	sumit = 0.0
	sig = []
	
	for i in range(len(p)):
		clus = 0.0
		for k in range(len(pi)):
			clus += pi[k]*mvnorm.pdf(p[i],mu[k],si[k])
		sumit += math.log(clus)
	if round(old,5) != round(sumit,5):
		return True,sumit
	else:
		return False,sumit


	pass

def init_mean(points,k):
	centers = []
	for i in range(0,k):
		center = random.choice(points)
		centers.append(center)
	return centers
def gmm(fread):
	data = file.read(file(fread)).splitlines()
	points = get_points(data)
	K = 3
	k_points = copy.deepcopy(points)
	#center,classes = assign_cluster(copy.copy(k_points),K)
	output_log = []
	final_center = []
	final_cov = []
	final_prior = []
	final_log = []
	best  = float('-Inf')
	for it in range(0,5):
		centers,classes = assign_cluster(copy.copy(k_points),K)
		center  = init_mean(copy.copy(k_points),K)
		mu = center
		pi = []
		cov_l = []
		prob = []
		p  = to_numpy(points)
		for i in classes:
			l = to_numpy(classes[i])
			pi.append(float(len(classes[i]))/float(len(points)))
			cov_l.append(np.identity(2))
			prob.append(mvnorm.pdf(p,mu[i],cov_l[i]))
		res = True
		old = 0.0
		log_out = []
		while(res):
			ric = E_Step(pi,prob)
			pi,mu,sigma = M_Step(p,ric)
			prob = []
			for i in classes:
				prob.append(mvnorm.pdf(p,mu[i],sigma[i]))

			res,old=logit(pi,p,mu,sigma,old)
			log_out.append(old)

		final_center.append(mu)
		final_cov.append(sigma)
		final_prior.append(pi)
		final_log.append(old)
		if old > best:
			best = old 
			best_mu = mu
			best_sig = sigma
			best_ric = ric
			best_log = old
			best_pi = pi

		output_log.append(log_out)

	color = ['red','blue','green','yellow','brown']
	for length in range(len(final_center)):
		print '\n -----------------Run   '+str(length+1)+'-----------------'
		print '\n CLUSTER CENTER \n',final_center[length]
		print '\n CLUSTER COVARIANCE \n',final_cov[length]
		print '\n PRIOR PROBABILITY \n',final_prior[length]
		print '\n LOG LIKELIHOOD  \n',final_log[length]


	plt.figure(1)
	for out in output_log:
		x= []
		for i in range(len(out)):
			x.append(i)
		plt.plot(x,out,color = color[output_log.index(out)])
	plt.show()

	print '\n BEST LOG-LIKELIHOOD : ',best_log

	print '\n BEST CLUSTER CENTERS '

	for centre in range(len(best_mu)):
		print '\n CLUSTER CENTER FOR CLUSTER  '+str(centre+1)+'\n'
		print '\t X : ',best_mu[centre][0]
		print '\t Y : ',best_mu[centre][1]

	print '\n BEST CLUSTER CO-VARIANCE '

	for centre in range(len(best_sig)):
		print '\n COVARIANCE FOR CLUSTER  '+str(centre+1)+'\n'
		print best_sig[centre]

	print '\n BEST PRIOR PROBABILITY \n'

	for centre in range(len(best_sig)):
		print '\n PRIOR PROBABILITY FOR CLUSTER  '+str(centre+1)+' : ',best_pi[centre]


	
	print best_log
	classes = {}
	for i in range(3):
		classes[i] = []
	for i in range(len(best_ric)):
		classes[best_ric[i].tolist().index(max(best_ric[i]))].append(np.array(p[i]))
	plt.figure(i)
	for j in  range(3):
		classes[j] = np.array(classes[j])
		x,y = classes[j][:,0],classes[j][:,1]
		plt.scatter(x,y,color=color[j])
	plt.show()




	'''
	for k in range(0,K):
		l = to_numpy(classes[k])
		print np.var(l[:,0])
		cov = np.zeros((p.shape[1],p.shape[1]))
		for i in range(0,p.shape[1]):
			for j in range(0,p.shape[1]):
				cov[i][j] = calc_cov(np.concatenate((l[:,i],l[:,j]),axis =0),pi,mu,k)
		cov_l.append(cov)
	'''
	
	

def to_list(data):
	p = []
	for j in data:
		p.append(j.tolist())
	return p


#### KERNEL KMEANS 

def get_kernel_matrix(points):
	km = np.zeros((len(points),len(points)))
	for i in range(len(points)):
		for j in range(len(points)):
			a = points[i][0] * points[j][0]
			b = points[i][1] * points[j][1]
			c = pow(points[i][0],2) + pow(points[i][1],2)
			d = pow(points[j][0],2) + pow(points[j][1],2)
			km[i][j] = a + b + 6*c * 6*d
	return km

def kernel_dist(points,i,kernel_m,classes,cluster,weight):
	sum1 = kernel_m[i][i]
	sum2 = 0.0
	l = len(classes[cluster])
	cl = 0

	for p in classes[cluster]:
		sum2 += kernel_m[points.index((p))][i]
	sum2 = sum2/l
	sum3 = weight
	sum3 = sum3/(l*l)
	return sum1 - (2*sum2) + sum3

def get_cluster_k(points,i,kernel_m,classes,weights):
	dist = []
	for cl in classes:
		dist.append(kernel_dist(points,i,kernel_m,classes,cl,weights[cl]))
	return dist.index(min(dist))




def get_cluster_weights(kernel_m,points,classes,cluster):
	sum3 = 0.0
	for k in classes[cluster]:
		in1 = points.index(k)
		for m in classes[cluster]:
			in2 = points.index((m))
			sum3 += kernel_m[in1][in2]
	return sum3

def class_to_list(cl):
	classes = {}
	for i in cl:
		classes[i] = []
		for j in ((cl[i])):
			classes[i].append(j.tolist())
	return classes

def class_to_numpy(cl):
	classes = {}
	for i in cl:
		classes[i] = []
		for j in ((cl[i])):
			classes[i].append(np.array(j))
		classes[i] = np.array(classes[i])
	return classes

def kkmeans(fread):
	data = file.read(file(fread)).splitlines()
	points = get_points(data)
	center,classes = assign_cluster(points,2)
	kernel_m = get_kernel_matrix(points)
	check = True
	classes = class_to_list(copy.copy(classes))
	while check:
		new_classes = {}
		cluster_weight = []
		for cluster in [0,1]:
			cluster_weight.append(get_cluster_weights(kernel_m,points,classes,cluster))
		for i in range(len(points)):
			k = get_cluster_k(points,i,kernel_m,classes,cluster_weight)
			new_classes = add_to_class(new_classes,points[i],k)
		check = check_cluster(classes,new_classes,2)
		classes = new_classes
	color = ['blue','red','green']
	classes = class_to_numpy(classes)
	for j in range(2):
		x,y = classes[j][:,0],classes[j][:,1]
		plt.scatter(x,y,color=color[j])
	plt.show()


		





if __name__ =="__main__":
	kmeans('hw5_blob.csv')
	kmeans('hw5_circle.csv')
	gmm('hw5_blob.csv')
	kkmeans('hw5_circle.csv')
	
	