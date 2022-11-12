

function ChainRulesCore.rrule(::typeof(Base.:*), m::MaskedMatrix, x::AbstractVector{S} where S<:Real)
    y = m * x
    project_Masked = ProjectTo(m)
    function foo_mul_pullback(ȳ)
        f̄ = NoTangent()
        # f̄oo = Tangent{typeof(m)}(; w= m.mask .* (ȳ * x'), mask=ZeroTangent())
        f̄oo = MaskedMatrix(m.mask .* (ȳ * x'), m.mask)
        b̄ = @thunk((m.mask .* m.w)' * ȳ)
        return f̄, f̄oo, b̄
    end
    return y, foo_mul_pullback
end

function ChainRulesCore.rrule(::typeof(Base.:*), m::MaskedMatrix, x::AbstractMatrix{S} where S<:Real)
    y = m * x
    project_Masked = ProjectTo(m)
    function foo_mul_pullback(ȳ)
        f̄ = NoTangent()
        # f̄oo = Tangent{typeof(m)}(; w= m.mask .* (ȳ * x'), mask=ZeroTangent())
        f̄oo = MaskedMatrix(m.mask .* (ȳ * x'), m.mask)
        b̄ = @thunk((m.mask .* m.w)' * ȳ)
        return f̄, f̄oo, b̄
    end
    return y, foo_mul_pullback
end