

module HW_unconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, PyPlot, DataFrames

	"""
    `input(prompt::AbstractString="")`

    Read a string from STDIN. The trailing newline is stripped.

    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export maximize_like_grad, runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	srand(1234)
	function makeData(n=10000)
		beta = [ 1; 1.5; -0.5 ]
		x= [rand(n) rand(n) rand(n)]
		prob = cdf(Normal(),*(x, beta))
		r = rand(Uniform(),n)
		y = Int[]
		for i in 1:n
    	if r[i] < prob[i]
        push!(y,1)
    	else
        push!(y,0)
    	end
		end
		d = Dict("beta"=> beta, "n"=> n, "x" => x, "y" => y, "prob" => prob)
		return d
	end
	d = makeData()

	# log likelihood function at x
	# function loglik(betas::Vector,X::Matrix,y::Vector,distrib::UnivariateDistribution)
	function loglik(betas::Vector{Float64},d::Dict)
    prob = cdf(Normal(),*(d["x"], betas))
    l = 0
    for i in 1:d["n"]
      l = l + d["y"][i] * log(prob[i]) + (1-d["y"][i]) * log(1-prob[i])
    end
    return l
	end

	# gradient of the likelihood at x
	function grad!(betas::Vector,storage::Vector)
		N=length(d["y"])
		med=zeros(N)
		quotient=zeros(N)
		quotient1=zeros(N)
		for i=1:N
		  med[i]=(d["x"][i,:]'*betas)[1]
		  quotient[i]=pdf(Normal(0,1),med[i])/cdf(Normal(0,1),med[i])
		  quotient1[i]=pdf(Normal(0,1),med[i])/(1-cdf(Normal(0,1),med[i]))
		end

		storage[1]=-sum(d["y"][i]*d["x"][i,1]*quotient[i]-(1-d["y"][i])*d["x"][i,1]*quotient1[i] for i=1:N)
		storage[2]=-sum(d["y"][i]*d["x"][i,2]*quotient[i]-(1-d["y"][i])*d["x"][i,2]*quotient1[i] for i=1:N)
		storage[3]=-sum(d["y"][i]*d["x"][i,3]*quotient[i]-(1-d["y"][i])*d["x"][i,3]*quotient1[i] for i=1:N)
		return storage
	end

	# hessian of the likelihood at x
	function hessian!(betas::Vector,storage::Matrix)
		N=length(d["y"])
		med = zeros(N)
		q11 = zeros(N)
		q12 = zeros(N)
		q13 = zeros(N)
		q21 = zeros(N)
		q22 = zeros(N)
		q23 = zeros(N)
		deriv1 = zeros(N)
		deriv2 = zeros(N)
		deriv3 = zeros(N)
		for i=1:N
			med[i]=(d["x"][i,:]'*betas)[1]
			deriv1[i] = -(1/sqrt(2*pi)) * med[i] * d["x"][i,1] * exp(-med[i]^2/2)
			deriv2[i] = -(1/sqrt(2*pi)) * med[i] * d["x"][i,2] * exp(-med[i]^2/2)
			deriv3[i] = -(1/sqrt(2*pi)) * med[i] * d["x"][i,3] * exp(-med[i]^2/2)
			q11[i]=(deriv1[i] * cdf(Normal(),med[i])-(pdf(Normal(),med[i]))^2)/(cdf(Normal(),med[i])^2)
			q12[i]=(deriv2[i] * cdf(Normal(),med[i])-(pdf(Normal(),med[i]))^2)/(cdf(Normal(),med[i])^2)
			q13[i]=(deriv3[i] * cdf(Normal(),med[i])-(pdf(Normal(),med[i]))^2)/(cdf(Normal(),med[i])^2)
			q21[i]=(deriv1[i] * (1-cdf(Normal(),med[i])) + (pdf(Normal(),med[i]))^2 )/(1-cdf(Normal(),med[i]))^2
			q22[i]=(deriv2[i] * (1-cdf(Normal(),med[i])) + (pdf(Normal(),med[i]))^2 )/(1-cdf(Normal(),med[i]))^2
			q23[i]=(deriv3[i] * (1-cdf(Normal(),med[i])) + (pdf(Normal(),med[i]))^2 )/(1-cdf(Normal(),med[i]))^2
		end

		storage[1,1] = -sum(d["y"][i]*(d["x"][i,1])^2 * q11[i] - (1-d["y"][i])*(d["x"][i,1])^2*q21[i] for i=1:N)
		storage[1,2] = -sum(d["y"][i]*d["x"][i,1]*d["x"][i,2]*q12[i] - (1-d["y"][i])*d["x"][i,1]*d["x"][i,2]*q22[i] for i=1:N)
		storage[1,3] = -sum(d["y"][i]*d["x"][i,1]*d["x"][i,3]*q13[i] - (1-d["y"][i])*d["x"][i,1]*d["x"][i,3]*q23[i] for i=1:N)

		storage[2,1] = -sum(d["y"][i]*d["x"][i,2]*d["x"][i,1]*q11[i] - (1-d["y"][i])*d["x"][i,1]*d["x"][i,2]*q21[i] for i=1:N)
		storage[2,2] = -sum(d["y"][i]*(d["x"][i,2])^2 * q12[i] - (1-d["y"][i])*(d["x"][i,2])^2*q22[i] for i=1:N)
		storage[2,3] = -sum(d["y"][i]*d["x"][i,2]*d["x"][i,3]*q13[i] - (1-d["y"][i])*d["x"][i,2]*d["x"][i,3]*q23[i] for i=1:N)

		storage[3,1] = -sum(d["y"][i]*d["x"][i,3]*d["x"][i,1]*q11[i] - (1-d["y"][i])*d["x"][i,3]*d["x"][i,1]*q21[i] for i=1:N)
		storage[3,2] = -sum(d["y"][i]*d["x"][i,3]*d["x"][i,2]*q12[i] - (1-d["y"][i])*d["x"][i,3]*d["x"][i,2]*q22[i] for i=1:N)
		storage[3,3] = -sum(d["y"][i]*(d["x"][i,3])^2 * q13[i] - (1-d["y"][i])*(d["x"][i,3])^2*q23[i] for i=1:N)
	end


	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
	end

	"""
	standard errors
	"""
	function se(betas::Vector,d::Dict)
	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(x0=[0.8,1.0,-0.1],meth=:nelder_mead)
		result = optimize(beta -> -loglik(beta,d), x0, NelderMead())
		return result
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:bfgsd)
		result = optimize(betas -> -loglik(betas,d), grad!, x0, BFGS())
		return result
	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=:newton)
		result = optimize(betas -> -loglik(betas,d), grad!, hessian!, x0, Newton())
	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=:bfgs)
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value.
	function plotLike()
		d = makeData()
	  lik = Any[]
	  for i in -2:.1:2
	    push!(lik, loglik([1; 1.5; i],d))
	  end
	  #plotlyjs()
	  pl = plot(-2:1/10:2,lik, xticks = (-2:.5:2),layout=3,title="Changing beta3",label="Log-likelihood")
	  vline!(pl[1],[-.5],label="True parameter: beta1")

	  lik2 = Any[]
	  for i in -2:.1:2
	    push!(lik2, loglik([1; i; -.5],d))
	  end
	  plot!(pl[2],-2:1/10:2,lik2, xticks = (-2:.5:2),title="Changing beta2",label="Log-likelihood")
	  vline!(pl[2],[1.5],label="True parameter: beta2")

	  lik3 = Any[]
	  for i in -2:.1:2
	    push!(lik3, loglik([i; 1.5; -.5],d))
	  end
	  plot!(pl[3],-2:1/10:2,lik3, xticks = (-2:.5:2),title="Changing beta1",label="Log-likelihood")
	  vline!(pl[3],[1],label="True parameter: beta3")
		savefig("tereza-plotLike.png")
	end

	function plotGrad()

	end


	function runAll()
		println("Beginning HW 3")
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		#m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like: $(m1.minimum)")
		println("maximize_like maximizer: $(m1.minimizer)")
		println("maximize_like_grad: $(m2.minimum)")
		println("maximize_like_grad maximizer: $(m2.minimizer)")
		println("maximize_like_grad_hess: $(m3.minimum)")
		println("maximize_like_grad_hess maximizer: $(m3.minimizer)")
		println("Generating the plot of log-likelihood (saved as .png file):")
		plotLike()
		#println("maximize_like_grad_se: $m4")
		#println("")
		println("running tests:")
		include("/test/tests.jl")
		#println("")
		#ok = input("enter y to close this session.")
		#if ok == "y"
			#quit()
		#end
		println("End of HW_unconstrained")
	end
end
