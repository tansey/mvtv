##########################################################################################################################################
##########################################################################################################################################

#Functions for crisp R package to perform methods described in "Convex Regression with Interpretable Sharp Partitions"
#i.e., these are functions that will likely be included in the forthcoming crisp R package

##########################################################################################################################################
##########################################################################################################################################

library(Matrix)
library(MASS)

##########################################################################################################################################
#### MAIN CRISP FUNCTIONS
### functions in this section:
### crisp.onelambda: fits CRISP for a single lambda value
### crisp.helper: the workhorse function for fitting CRISP, calls crisp.onelambda for each value of lambda
### crisp: this is the function you should call to fit CRISP, it will in turn call the two functions above as needed
##########################################################################################################################################

crisp.onelambda = function(y, X, lambda, rho=0.1, e_abs=10^-4, e_rel=10^-3, varyrho=TRUE, initial.m=NULL, initial.u=NULL, initial.z=NULL, Q, A, z.shrink=NULL) {
	
	n = sqrt(ncol(A))
	
	#shrink Q and A if !is.null(z.shrink)
	if (!is.null(z.shrink)==T) {
		blocks = get.blocks(z=z.shrink, n=n)
		Q = sapply(blocks,colReduce,matrix=Q,simplify=T)
		A = sapply(blocks,colReduce,matrix=A,simplify=T)
	}
	
	converge = FALSE
	if (is.null(initial.m)) m = matrix(0, nrow=ncol(A), ncol=1) else m = initial.m
	if (is.null(initial.z)) z = matrix(0, nrow=nrow(A), ncol=1) else z = initial.z
	if (is.null(initial.u)) u = matrix(0, nrow=nrow(A), ncol=1) else u = initial.u
	n.iter = 0
	rho.old = rho

	indices = seq(1,nrow(A),by=n)
	
	#matrices used later
	tmp = crossprod(Q) + rho * crossprod(A)
	crossprodQ = crossprod(Q)
	crossprodQy = crossprod(Q,y)
	QRmat = qr(tmp)
		
	while (converge==F) {
		#step 1: update m
		if (rho.old!=rho) {
			tmp = crossprodQ + rho * crossprod(A)
			QRmat = qr(tmp)
		}
		m = qr.coef(QRmat, crossprodQy + rho * crossprod(A, z-u))
		Am = A %*% m
		
		#step 2: update z
		z.old = z
		z = matrix(as.vector(sapply(indices,	 function(index, n, vec, lambda, rho) 
			update.l2(vec[index:(index+n-1)], lambda, rho), lambda=lambda, rho=rho, n=n, vec= Am+u)),ncol=1)
	
		#step 3: update u
		u = Am + u - z
				
		#check convergence
		n.iter = n.iter + 1
		#primal (r) and dual (s) residuals
		r = Am - z
		s = rho * crossprod(A, z.old - z)
		r.norm = sqrt(sum(r^2)); s.norm = sqrt(sum(s^2))
		#thresholds
		e_primal = sqrt(nrow(A)) * e_abs + e_rel * max(c(sqrt(sum((Am)^2)),sqrt(sum(z^2))))
		e_dual = n * e_abs + e_rel * sqrt(sum((rho*crossprod(A, u))^2))
		if ((r.norm <= e_primal) & (s.norm <= e_dual) & n.iter>2) converge = TRUE
		
		#update rho (and u) when allowing rho to vary
		rho.old = rho
		
		if (varyrho==T) {
			if (r.norm > (10*s.norm)) {
				rho = 2*rho; u = u/2
			} else if (s.norm > (10*r.norm)) {
				rho = rho/2; u = 2*u
			}
		}		
	}
	
	obj = calc.obj(y=y, X=X, m=m, lambda=lambda, Q=Q, A=A)
	
	if (!is.null(z.shrink)==T) {
		m.shrunk = m
		m = rep(NA, n^2)
		for (i in 1:length(blocks)) m[blocks[[i]]] = m.shrunk[i] 
	}
	
	return(list(M=matrix(m, nrow=n), z=z, u=u, n.iter=n.iter, obj.value=obj, y=y, X=X, lambda=lambda, rho=rho, e_abs=e_abs, e_rel=e_rel, z.shrink=z.shrink))
}

crisp.helper = function(y, X, lambda.seq, rho=0.1, e_abs=10^-4, e_rel=10^-3, varyrho=TRUE, z.shrink=NULL, initial.M.list=NULL, initial.u.mat=NULL, initial.z.mat=NULL, A, Q) {
	#make sure lambda.seq is decreasing
	lambda.seq = sort(lambda.seq, decreasing=TRUE)

	#initialize
	M.hat.list = vector("list",length(lambda.seq))
	z.hat.mat <- u.hat.mat <- matrix(NA, nrow=nrow(A), ncol=length(lambda.seq))
	n.iter.vec <- obj.vec <- rep(NA, length(lambda.seq))

	#fit model
	for (i in 1:length(lambda.seq)) {	
		print(sprintf("Current lambda: %f", lambda.seq[i]))
		if (is.null(z.shrink)) {		
			if (i==1) {
				out = crisp.onelambda(y=y, X=X, lambda=lambda.seq[i], rho=rho, e_abs=e_abs, e_rel=e_rel, varyrho=varyrho, 
					initial.m=NULL, initial.z=NULL, initial.u=NULL, Q=Q, A=A)
			} else {
				out = crisp.onelambda(y=y, X=X, lambda=lambda.seq[i], rho=out$rho, e_abs=e_abs, e_rel=e_rel, varyrho=varyrho, 
				initial.m=as.vector(M.hat.list[[i-1]]), initial.z=z.hat.mat[,i-1], initial.u=u.hat.mat[,i-1], Q=Q, A=A)
			}
		} else {	
			blocks = get.blocks(z=z.shrink[,i], n=n)
			initial.m = matrix(NA, nrow=length(blocks), ncol=1)
			for (j in 1:length(blocks)) initial.m[j] = as.vector(initial.M.list[[i]])[blocks[[j]][1]]	
			out = crisp.onelambda(y=y, X=X, lambda=lambda.seq[i], rho=rho, e_abs=e_abs, e_rel=e_rel, varyrho=varyrho, 
				initial.m=initial.m, initial.z=initial.z.mat[,i], initial.u=initial.u.mat[,i], Q=Q, A=A, z.shrink=z.shrink[,i])
		}
		
		M.hat.list[[i]] = out$M
		z.hat.mat[,i] = as.vector(out$z)
		u.hat.mat[,i] = as.vector(out$u)
		n.iter.vec[i] = out$n.iter; obj.vec[i] = out$obj.value
	}
	
	return(list(M.hat.list=M.hat.list,z.hat.mat=z.hat.mat,u.hat.mat=u.hat.mat,n.iter.vec=n.iter.vec, obj.vec=obj.vec, y=y, X=X, lambda.seq=lambda.seq, rho=rho, e_abs=e_abs, e_rel=e_rel, z.shrink=z.shrink))
}

findIntervalOverlaps = function(x, vec, rightmost.closed = FALSE, all.inside = FALSE) {
    uniquevec = unique(vec)
    result = rep(0, length(x))
    for (i in 1:length(x)){
        block = findInterval(x[i], sort(uniquevec), rightmost.closed = rightmost.closed, all.inside = all.inside)
        result[i] = which.min(vec == uniquevec[block])
    }
    return(result)
}

#fits CRISP for a decreasing sequence of lambda values
##inputs:
# y: n-vector containing the response
# X: a n x 2 matrix with the features
# lambda.min.ratio: smallest value for lambda.seq, as a fraction of the maximum lambda value, which is the data-derived 
#                   smallest value for which all estimated functions are zero. The default is 0.01.
# n.lambda: the number of lambda values to consider - the default is 50.
# lambda.seq: a user-supplied sequence of positive lambda values to consider. The typical usage is to calculate 
#             lambda.seq using lambda.min.ratio and n.lambda, but providing lambda.seq overrides this. If provided,
#             lambda.seq should be a decreasing sequence of values, since CRISP relies on warm starts for speed.
#             Thus fitting the model for a whole sequence of lambda values is often faster than fitting for a single 
#             lambda value.
# e_abs and e_rel: using in the stopping criterion for our ADMM algorithm, discussed in Appendix C.2 of CRISP paper
# rho: penalty parameter for ADMM
# varyrho: should rho be varied from iteration to iteration? discussed in Appendix C.3 of CRISP paper
# double.run: the initial complete run of our ADMM algorithm will yield sparsity in z_{1i} and z_{2i}, but not 
#             necessarily exact equality of the rows and columns of M.hat. If double.run is TRUE, then the algorithm
#             is run a second time to obtain M.hat with exact equality of the appropriate rows and columns. This issue
#             is discussed further in Appendix C.4 of CRISP paper.
# q: desired dimension of M hat, if NULL then uses q=n, 
#     we recommend using q<=100 as higher values take longer to fit and provide an unneeded amount of granularity  
##outputs:
# returns a list with the inputs, along with:
# M.hat.list: a list of length n.lambda giving M.hat for each value of lambda.seq
# num.blocks: a vector of length n.lambda giving the number of blocks in M.hat for each value of lambda.seq
# obj.vec: a vector of length n.lambda giving the value of the objective of Eqn (4) in CRISP paper for each value of lambda.seq

crisp = function(y, X, lambda.min.ratio = 0.01, n.lambda = 50, lambda.seq = NULL, rho=0.1, e_abs=10^-4, e_rel=10^-3, varyrho=TRUE, double.run=FALSE, q=NULL) {
	n = length(y)
	if (is.null(q)) q = n #default: use q=n (original CRISP proposal)
    if (is.null(lambda.seq)) {
        max.lam = max.lambda(X=X, y=y)
        lambda.seq = exp(seq(log(max.lam), log(max.lam * lambda.min.ratio), len = n.lambda))
    }

	#checks
	if (length(y)!=nrow(X)) stop("The length of 'y' must equal the number of rows of 'x'")
	if (length(lambda.seq)==1) stop("Provide a sequence of decreasing values for 'lambda.seq'")
	if (min(lambda.seq)<=0) stop("Values in 'lambda.seq' must be positive")
	if (e_abs<=0 | e_rel<=0) stop("Values for 'e_abs' and 'e_rel' must be positive")

	if (q!=n) {
		n = q
		q.seq = c(0, 1:(q-1)/q, 1)
		block.X1 = findIntervalOverlaps(X[,1], quantile(X[,1], q.seq, type=8), all.inside=T)
		block.X2 = findIntervalOverlaps(X[,2], quantile(X[,2], q.seq, type=8), all.inside=T)
		blocks = block.X1 + n * (block.X2 - 1)
		Q = sparseMatrix(i=1:nrow(X), j=blocks, dims=c(nrow(X),n^2))
		Q = as(Q, "dgCMatrix") # TEMP
	} else {
		Q = get.Q(n=n, X=X, sparse=T)
	}

	A = get.A(n=n, sparse=T)

	out.unshrunk = crisp.helper(y=y, X=X, lambda.seq=lambda.seq, rho=rho, e_abs=e_abs, e_rel=e_rel, varyrho=varyrho, A=A, Q=Q)
	#n.iter1 = out.unshrunk$n.iter.vec

	if (double.run==TRUE) {
			
		out.shrunk = crisp.helper(y=y, X=X, lambda.seq=lambda.seq, rho=rho, e_abs=e_abs, e_rel=e_rel, 
			varyrho=varyrho, z.shrink=out.unshrunk$z.hat.mat, initial.M.list=out.unshrunk$M.hat.list, 
			initial.u.mat=out.unshrunk$u.hat.mat, initial.z.mat=out.unshrunk$z.hat.mat, A=A, Q=Q)

		num.blocks = apply(out.shrunk$z.hat.mat, 2, function(z, n) length(get.blocks(z=z, n=n)), n=n)
		M.hat.list = out.shrunk$M.hat.list
		obj.vec = out.shrunk$obj.vec
		
	} else {
		
		num.blocks = apply(out.unshrunk$z.hat.mat, 2, function(z, n) length(get.blocks(z=z, n=n)), n=n)
		M.hat.list = out.unshrunk$M.hat.list
		obj.vec = out.unshrunk$obj.vec	
	}
	
	return(list(lambda.seq=lambda.seq, M.hat.list=M.hat.list, num.blocks=num.blocks, obj.vec=obj.vec, y=y, X=X, rho=rho, e_abs=e_abs, e_rel=e_rel, varyrho=varyrho, double.run=double.run))		
}

##########################################################################################################################################
#### METHODS FOR CRISP OBJECT
##########################################################################################################################################

get.cell = function(block.X1, block.X2, ntilde) {
	
	index = block.X1 + ntilde * (block.X2 - 1)
	return(index)
}

#predict to have same value as closest point in grid (observed or unobserved)
predict.crisp = function(object, new.X, lambda.index, ...) {
	
	x1.sort = sort(object$X[,1]); x2.sort = sort(object$X[,2])

	ntilde = nrow(object$M.hat.list[[1]])
		q.seq = c(0, 1:(ntilde-1)/ntilde, 1)
	block.X1 = findIntervalOverlaps(new.X[,1], quantile(x1.sort, q.seq, type=8), all.inside=T)
	block.X2 = findIntervalOverlaps(new.X[,2], quantile(x2.sort, q.seq, type=8), all.inside=T)
	
	closest = sapply(1:nrow(new.X), function(block.X1, block.X2, ntilde, i) get.cell(block.X1[i], block.X2[i], ntilde), ntilde=ntilde, block.X1=block.X1, block.X2=block.X2)
	y.hat.new = as.vector(object$M.hat.list[[lambda.index]])[closest]
	
	return(y.hat.new)
}

##########################################################################################################################################
#### OTHER FUNCTIONS
##########################################################################################################################################

mse = function(y, y.hat) {
	sum((y - y.hat)^2)/length(y)
}

closest.index = function(x1, x2, X) {
	dist.sq = (X[,1] - x1)^2 + (X[,2] - x2)^2
	index = which(min(dist.sq)==dist.sq)[1]
	return(index)
}

max.lambda = function(X, y, A=NULL, Q=NULL) {
	
	n = length(y)
	if (is.null(A)) A = get.A(n=n, sparse=F)
	if (is.null(Q)) Q = get.Q(n=n, X=X, sparse=F)
	indices = seq(1,2*n*(n-1),by=n)
	
	stack = rbind(A,Q)
	M = -cbind(matrix(0,nrow=n,ncol=2*n*(n-1)),diag(n)) + Q %*% ginv(stack)
	w = ginv(M) %*% y
	u_A = (cbind(diag(2*n*(n-1)), matrix(0,nrow=2*n*(n-1),ncol=n)) - A %*% ginv(stack)) %*% w
	lambda = max(sapply(indices, function(index, y) sqrt(sum((y[index:(index+n-1)])^2)), y=u_A))

	return(lambda)
}

get.A = function(n, sparse=T) {
	
	index1 = rep(1:(n-1),each=n)
	index2 = rep(1:n,(n-1))
	A.i = c((index1-1)*n+index2, n*(n-1)+(index1-1)*n+index2, 
		(index1-1)*n+index2, n*(n-1)+(index1-1)*n+index2)
	A.j = c((index2-1)*n+index1, (index1-1)*n+index2, 
		(index2-1)*n+index1+1, index1*n+index2)
	A.x = c(rep(1,2*n*(n-1)), rep(-1,2*n*(n-1)))

	if (sparse==T) {
		A = sparseMatrix(i=A.i, j=A.j, x=A.x, dims=c(2*n*(n-1),n^2))
	} else if (sparse==F) {
		A = matrix(0, nrow=2*n*(n-1), ncol=n^2)
		for (l in 1:length(A.i)) A[A.i[l],A.j[l]] = A.x[l]
	}
	return(A)
}

get.Q = function(n, X, sparse=T) {
	
	Q.i = 1:n; Q.j = rank(X[,1]) + (rank(X[,2])-1)*n
	if (sparse==T) {
		Q = sparseMatrix(i=Q.i, j=Q.j, dims=c(n,n^2))
		Q = as(Q, "dgCMatrix") # TEMP
	} else if (sparse==F) {
		Q = matrix(0, nrow=n, ncol=n^2)
		for (l in Q.i) Q[Q.i[l],Q.j[l]] = 1	
	}
	return(Q)
}

get.blocks = function(z,n) {

	indices = seq(1,n*(n-1),by=n)

	rows.same = as.vector(sapply(indices, function(index, vec, n) 
		ifelse(sum((vec[index:(index+n-1)])^2)==0,1,0), 
		vec=z[1:(n*(n-1))], n=n))
	
	cols.same = as.vector(sapply(indices, function(index, vec, n) 
		ifelse(sum((vec[index:(index+n-1)])^2)==0,1,0), 
		vec=z[(1+n*(n-1)):(2*n*(n-1))], n=n))

	blocks = vector("list")
	elements = matrix(1:(n^2),nrow=n)
	col = 1

	while (col<=n) {

		#check which columns are the same
		k_c = 0
		while(cols.same[col+k_c]!=0 & (col+k_c)<n) k_c = k_c + 1
	
		row = 1
		
		while (row<=n) {
		
			#check which rows are the same
			k_r = 0
			while(rows.same[row+k_r]!=0 & (row+k_r)<n) k_r = k_r + 1
			
			#add to 'blocks'
			blocks[[length(blocks)+1]] = as.vector(elements[row:(row+k_r),col:(col+k_c)])
	
			#adjust row counter
			row = row + k_r + 1
		}

		#adjust column counter
		col = col + k_c + 1
	}
	
	return(blocks)
}

colReduce = function(matrix,indices) {
	if(length(indices)==1) {
		col = matrix[,indices]
	} else col = rowSums(matrix[,indices])
	return(col)
}

update.l2 = function(vec, lambda, rho) {
	norm.vec = sqrt(sum(vec^2))
	z_chunk = vec * max(c(1 - lambda/(rho*norm.vec), 0))
	return(z_chunk)
}

calc.obj = function(y, X, m, lambda, Q=NULL, A=NULL) {

	n = length(y)
	indices = seq(1,2*n*(n-1),by=n)
	
	if (is.null(Q)) Q = get.Q(n=n, X=X, sparse=TRUE)
	if (is.null(A)) A = get.A(n=n, sparse=TRUE)

	penalty = Reduce('+',sapply(indices, function(index, vec) sqrt(sum((vec[index:(index+n-1)])^2)), vec=A%*%m))
	obj = 0.5 * sum((y - Q %*% m)^2) + lambda * penalty
	
	return(obj)
}

##########################################################################################################################################
#### FUNCTIONS TO GENERATE DATA
##########################################################################################################################################

mean.model = function(x1,x2,scenario) {
	
	if (scenario==1) {
		#additive model
		if (x1*x2 > 0) mean = 2*sign(x1) else mean = 0
	} else if (scenario==2) {
		#interaction model
		if (x1*x2 > 0) mean = -2 else mean = 2
		mean = mean/sqrt(2)
	} else if (scenario==3) {
		#tetris model
		if (x1<(-5/6)) {
			if (x2<(-1.25)) mean = -3 else mean = 1
		} else if (x1<5/6) {
			if (x2<0) mean = -2 else mean =2
		} else {
			if (x2<1.25) mean = -1 else mean = 3
		}
		mean = mean/sqrt(5/3)
	} else if (scenario==4) {
		#smooth model
		d = 3
		mean  = 10/(((x1-2.5)/d)^2 + ((x2-2.5)/d)^2 + 1) + 10/(((x1+2.5)/d)^2 + ((x2+2.5)/d)^2 + 1)
		mean = mean - 8.476032; mean = mean/ sqrt(1.936208/2)
	} else if (scenario==5) {
		if ((x1<(-0.83) & x2<(-0.83)) |  (x1>(-0.83) & x2>(-0.83))) mean = -2 else mean = 2
		mean = mean + 0.2048
		mean = mean/sqrt(3.958057/2)
	}
	return(mean)
}

#function to generate simulated data used in CRISP paper
##inputs:
# n: number of observatons
# scenario: either 1 (additive model), 2 (interaction model), 3 ('tetris' model), 4 (smooth model)
# noise: SD of normal noise that is added to signal
# X: n x 2 covariate matrix (optional)
##outputs:
#returns a list including
# X: n x 2 covariate matrix
# y: n-vector with the response
sim.data = function(n, scenario, noise=1, X=NULL) {
	
	if (is.null(X)) X = matrix(runif(n=n*2,min=-2.5,max=2.5), nrow=n, ncol=2)	
	y = sapply(1:nrow(X),function(index,matrix,scenario) mean.model(matrix[index,1],matrix[index,2],scenario),matrix=X,scenario=scenario) + rnorm(n,sd=noise)
	
	return(list(X=X, y=y))
}