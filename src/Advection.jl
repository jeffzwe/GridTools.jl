include("Metric.jl")
using .GridTools

@field_operator function with_boundary_values(
    lowwer::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    interior::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    upper::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    level_indices::Field{<:Integer, 1, Tuple{K_}, <:Tuple},
    num_level::Integer
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}

    return where(level_indices .== 0, lowwker, where(level_indices .== num_level-1, upper, interior))
end

@field_operator function nabla_z(
    psi::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}, 
    level_indices::Field{<:Integer, 1, Tuple{K_}, <:Tuple}, 
    num_level::Integer
    )
    return with_boundary_values(
        psi(Koff(2)) - psi(Koff(1)),
        psi(Koff(2)) - psi(Koff()),
        psi(Koff(1)) - psi(Koff(1)),
        level_indices, num_level
    )
end

@field_operator function advector_in_edges(
    vel_x::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_y::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple}
    )::Tuple{Field, Field}
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(E2V(1)) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V(1)) + pole_bc * vel_y(E2V[1]))
    return vel_edges_x, where(pole_edge_mask, 0.0, vel_edges_y)
end

@field_operator function advector_normal(
    vel_x::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_y::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple},
    dual_face_normal_weighted_x::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple},
    dual_face_normal_weighted_y::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 .* (vel_x(E2V(1)) .+ pole_bc .* vel_x(E2V(2)))
    vel_edges_y = 0.5 .* (vel_y(E2V(1)) .+ pole_bc .* vel_y(E2V(2)))
    vel_edges_y = where(pole_edge_mask, 0.0, vel_edges_y)
    # vel_edges_x = where(pole_edge_mask, 0.0, vel_edges_x)
    return vel_edges_x .* dual_face_normal_weighted_x .+ vel_edges_y .* dual_face_normal_weighted_y
end

@field_operator function upstream_flux(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_x::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_y::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple},
    dual_face_normal_weighted_x::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple},
    dual_face_normal_weighted_y::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    vel_x_face, vel_y_face = advector_in_edges(vel_x, vel_y, pole_edge_mask)
    wnv = vel_x_face .* dual_face_normal_weighted_x .+ vel_y_face .* dual_face_normal_weighted_y
    return where(wnv .> 0.0, rho(E2V(1)) .* wnv, rho(E2V(2)) .* wnv)
end

@field_operator function upwind_flux(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    veln::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    return where(veln .> 0.0, rho(E2V(1)) .* veln, rho(E2V(2)) .* veln)
end

@field_operator function centered_flux(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    veln::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    return (
        0.5 .* veln .* (rho(E2V(2)) + rho(E2V(1)))
    )  # todo(ckuehnlein): polar flip for u and v transport later
end

@field_operator function pseudo_flux(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    veln::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    grg::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    cfluxdiv::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    dt::AbstractFloat
    )::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    return 0.5 .* abs.(veln) .* (rho(E2V(2)) - rho(E2V(1))) .- dt.* veln .* 0.5 .* (
        (cfluxdiv(E2V(2)) .+ cfluxdiv(E2V(1))) ./ (grg(E2V(2)) .+ grg(E2V(1)))
    )
end

@field_operator function limit_pseudo_flux(
    flux::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    cn::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    cp::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    )::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple}
    return max.(0.0, flux) .* min.(1.0, min.(cp(E2V(2)), cn(E2V(1)))) + min.(
        0.0, flux
    ) .* min.(1.0, min.(cn(E2V(2)), cp(E2V(1))))
end

@field_operator function flux_divergence(
    flux::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    vol::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    return 1.0 ./ (vol .* gac) .* neighbor_sum(flux(V2E()) .* dual_face_orientation, axis=V2EDim)
end

@field_operator function nonoscoefficients_cn(
        psimin::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
        psi::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
        flux::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
        vol::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
        gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
        dt::AbstractFloat,
        eps::AbstractFloat,
        dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    zrhout = (1.0 ./ vol) .* neighbor_sum(
        (
            max.(0.0, flux(V2E())) .* max.(0.0, dual_face_orientation)
            .+ min.(0.0, flux(V2E())) .* min.(0.0, dual_face_orientation)
        ),
        axis=V2EDim,
    )
    return (psi .- psimin) .* gac ./ (zrhout .* dt .+ eps)
end

@field_operator function nonoscoefficients_cp(
    psimax::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    psi::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    flux::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    vol::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    dt::AbstractFloat,
    eps::AbstractFloat,
    dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    zrhin = (1.0 ./ vol) .* neighbor_sum(
        -min.(0.0, flux(V2E())) .* max.(0.0, dual_face_orientation)
        - max.(0.0, flux(V2E())) .* min.(0.0, dual_face_orientation),
        axis=V2EDim,
    )
    return (psimax .- psi) .* gac ./ (zrhin .* dt .+ eps)
end

@field_operator function local_min(
    psi::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    return min.(psi, min_over(psi(V2V()), axis=V2VDim))
end

@field_operator function local_max(
    psi::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    return max.(psi, max_over(psi(V2V()), axis=V2VDim))
end

@field_operator function update_solution(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    flux::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    dt::AbstractFloat,
    vol::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    return rho .- dt ./ (vol .* gac) .* neighbor_sum(flux(V2E()) .* dual_face_orientation, axis=V2EDim)
end

@field_operator function advect_density(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    dt::AbstractFloat,
    vol::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    vel_x::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_y::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple},
    dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple},
    dual_face_normal_weighted_x::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple},
    dual_face_normal_weighted_y::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    
    veln = advector_normal(vel_x, vel_y, pole_edge_mask, dual_face_normal_weighted_x, dual_face_normal_weighted_y,)

    flux = upwind_flux(rho, veln)
    rho = update_solution(rho, flux, dt, vol, gac, dual_face_orientation)

    cflux = centered_flux(rho, veln)
    cfluxdiv = flux_divergence(cflux, vol, gac, dual_face_orientation)

    pseudoflux = pseudo_flux(rho, veln, gac, cfluxdiv, dt)
    rho = update_solution(rho, pseudoflux, dt, vol, gac, dual_face_orientation)

    return rho
end

@field_operator function mpdata_program(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    dt::AbstractFloat,
    eps::AbstractFloat,
    gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    vel_x::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_y::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_z::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple},
    dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple},
    dual_face_normal_weighted_x::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple},
    dual_face_normal_weighted_y::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple},
    tmp_vertex_1::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    tmp_vertex_2::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    tmp_vertex_3::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    tmp_vertex_4::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    tmp_vertex_5::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    tmp_vertex_6::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    tmp_edge_1::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    tmp_edge_2::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    tmp_edge_3::Field{<:AbstractFloat, 2, Tuple{Edge_, K_}, <:Tuple},
    )

    tmp_edge_1 = advector_normal(
        vel_x,
        vel_y,
        pole_edge_mask,
        dual_face_normal_weighted_x,
        dual_face_normal_weighted_y
    )

    tmp_edge_2 = upwind_flux(rho0, tmp_edge_1)
    tmp_vertex_1 = update_solution(
        rho0,
        tmp_edge_2,
        dt,
        vol,
        gac,
        dual_face_orientation
    )

    tmp_vertex_3 = local_min(rho0)
    tmp_vertex_4 = local_max(rho0)

    tmp_edge_2 = centered_flux(tmp_vertex_1, tmp_edge_1)
    tmp_vertex_2 = flux_divergence(tmp_edge_2, vol, gac, dual_face_orientation)
    tmp_edge_2 = pseudo_flux(tmp_vertex_1, tmp_edge_1, gac, tmp_vertex_2, dt)

    tmp_vertex_5 = nonoscoefficients_cn(
        tmp_vertex_3,
        tmp_vertex_1,
        tmp_edge_2,
        vol,
        gac,
        dt,
        eps,
        dual_face_orientation
    )

    tmp_vertex_6 = nonoscoefficients_cp(
        tmp_vertex_4,
        tmp_vertex_1,
        tmp_edge_2,
        vol,
        gac,
        dt,
        eps,
        dual_face_orientation
    )

    tmp_edge_3 = limit_pseudo_flux(tmp_edge_2, tmp_vertex_5, tmp_vertex_6)

    # corresponds to rho1 in the python version
    return update_solution(
        tmp_vertex_1,
        tmp_edge_3,
        dt,
        vol,
        gac,
        dual_face_orientation
    )
end


@field_operator function upwind_scheme(
    rho::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    dt::AbstractFloat,
    vol::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    gac::Field{<:AbstractFloat, 1, Tuple{Vertex_}, <:Tuple},
    vel_x::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_y::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    vel_z::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple},
    pole_edge_mask::Field{Bool, 1, Tuple{Edge_}, <:Tuple},
    dual_face_orientation::Field{<:AbstractFloat, 2, Tuple{Vertex_, V2EDim_}, <:Tuple},
    dual_face_normal_weighted_x::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple},
    dual_face_normal_weighted_y::Field{<:AbstractFloat, 1, Tuple{Edge_}, <:Tuple}
    )::Field{<:AbstractFloat, 2, Tuple{Vertex_, K_}, <:Tuple}
    vn = advector_normal(vel_x, vel_y, pole_edge_mask, dual_face_normal_weighted_x, dual_face_normal_weighted_y, offset_provider = offset_provider)

    flux = upwind_flux(rho, vn, offset_provider = offset_provider)
    rho = rho .- dt ./ (vol .* gac) .* neighbor_sum(flux(V2E()) .* dual_face_orientation, axis=V2EDim)
    return rho
end

