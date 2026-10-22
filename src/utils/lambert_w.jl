"""
    lambert_w0_from_log(log_argument::Real)

The principal branch W₀ of the Lambert W function, which solves w * exp(w) = z for z ≥ 0, given the log of its argument z.
Taking the log of the argument means that z itself never has to be represented, so it can be far above the largest representable number.
Uses Halley's method, which converges to machine precision in a few iterations. The iteration is written in terms of the residual w - exp(log(z) - w), so that no intermediate value can overflow.
"""
function lambert_w0_from_log(log_argument::Real)

    #For very small arguments, W₀(z) = z to machine precision
    if log_argument < -36
        return exp(log_argument)
    end

    #Initial guess log(z) - log(log(1 + z)), which is close to the solution for all arguments
    w = log_argument - log(log1pexp(log_argument))

    #Halley iterations
    for _ = 1:8

        #The residual, which is zero at the solution
        residual = w - exp(log_argument - w)

        #Halley step
        w -= 2 * residual * (1 + w) / (2 * (1 + w)^2 - residual * (2 + w))
    end

    return w
end
