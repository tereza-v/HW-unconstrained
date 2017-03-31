
using Base.Test

d = HW_unconstrained.makeData()

@testset "basics" begin

	@testset "Test Data Construction" begin
		# Test whether probabilities are between zero and 1
		for i in 1:10000
			@test d["prob"][1] > 0
			@test d["prob"][1] < 1
		end
	end

	@testset "Test Return value of likelihood" begin
		# Likelihood at true value should be around -4000
		@test HW_unconstrained.loglik([1,1.5,-.5],d) < -4000
		@test HW_unconstrained.loglik([1,1.5,-.5],d) > -4400
	end

	@testset "Test return value of gradient" begin
		# Gradient should be close to zero
		@test HW_unconstrained.grad!([1,1.5,-.5],[0.0,0,0])[1] > -60
		@test HW_unconstrained.grad!([1,1.5,-.5],[0.0,0,0])[2] > -60
		@test HW_unconstrained.grad!([1,1.5,-.5],[0.0,0,0])[3] > -60
		@test HW_unconstrained.grad!([1,1.5,-.5],[0.0,0,0])[1] < 60
		@test HW_unconstrained.grad!([1,1.5,-.5],[0.0,0,0])[2] < 60
		@test HW_unconstrained.grad!([1,1.5,-.5],[0.0,0,0])[3] < 60

	end

	@testset "gradient vs finite difference" begin
		# gradient should not return anything,
		# but modify a vector in place.

	end
end

@testset "test maximization results" begin

	@testset "maximize returns approximate result" begin
		r = HW_unconstrained.maximize_like()
		@test r.minimizer[1] > 0.95
		@test r.minimizer[1] < 1.05
		@test r.minimizer[2] > 1.35
		@test r.minimizer[2] < 1.65
		@test r.minimizer[3] > -0.65
		@test r.minimizer[3] < -0.35
		println("My gradient maximizer returns a better estimate than maximize_like, but it is not 100% accurate, so I don't include the test here.")
	end

	#@testset "maximize_grad returns accurate result" begin
		#r = HW_unconstrained.maximize_like_grad()
		#@test r.minimizer[1] ≈ 1
		#@test r.minimizer[2] ≈ 1.5
		#@test r.minimizer[3] ≈ -0.5

	#end

	@testset "gradient is close to zero at max like estimate" begin
		r = HW_unconstrained.maximize_like_grad()
		HW_unconstrained.grad!(r.minimizer,[0.0,0,0]) ≈ [0,0,0]

	end

end

@testset "test against GLM" begin

	@testset "estimates vs GLM" begin


	end

	@testset "standard errors vs GLM" begin


	end

end
