"""The configuration record saved beside every run's samples.

A results.pt cannot be read back without this. The same lambda means a
different tradeoff under a different sense or constraint, so an arm name alone
does not identify what was optimized -- the objective definition has to travel
with the samples.
"""


def build_run_config(args, setup, spec, oracle_path, path=None) -> dict:
    """Everything needed to interpret, or reproduce, this run."""
    threshold = spec.constraint_threshold
    return {
        "dataset":         str(args.dataset),
        "esm_model":       args.esm_model,
        "epochs":          args.epochs,
        "samples":         args.samples,
        "max_length":      args.max_length,
        "decode":          "mut_budget" if args.mut_budget is not None else "min_polar",
        "mut_budgets":     list(setup.budgets) if setup.budgets else None,
        "interpolant":     args.interpolant,
        "path_geometry":   path.geometry if path else args.path_geometry,
        "time_schedule":   path.schedule if path else args.time_schedule,
        "sample_schedule": path.sample_schedule if path else args.sample_schedule,
        "path_scale":      path.scale if path else args.path_scale,
        "steps":           args.steps,
        "coupling":        args.coupling,
        "coupling_beta":   args.coupling_beta,
        "coupling_columns": (list(args.coupling_columns)
                             if args.coupling_columns else None),
        "mut_budget":      setup.primary,
        "decode_temperature": args.decode_temperature,
        "exact_mutations":    args.exact_mutations,
        "restrict_support":   (str(args.restrict_support)
                               if args.restrict_support else None),
        "support_level":      args.support_level if args.restrict_support else None,
        "freeze_positions":   list(setup.frozen),
        # Objective definition.
        "property_names":     list(spec.prop_names),
        "n_objectives":       spec.n_obj,
        "objective_senses":   list(spec.senses),
        "setpoints_raw":      [float(v) for v in spec.setpoint_raw] or None,
        "setpoints_given":    list(args.setpoint) if args.setpoint else None,
        "setpoint_percentile": bool(args.setpoint_percentile),
        "minimize":           list(args.minimize) if args.minimize else None,
        "constraint_property": args.constraint_property,
        "constraint_delta":   args.constraint_delta if args.constraint_rho else None,
        "constraint_rho":     args.constraint_rho or None,
        "constraint_threshold_standardized": (float(threshold)
                                              if threshold is not None
                                              else None),
        "min_polar":       args.min_polar,
        "reference":       setup.reference,
        "anchor_strength": args.anchor_strength,
        "batch_size":      args.batch_size,
        "hidden":          args.hidden,
        "arch":            args.arch,
        "predict":         args.predict,
        "ema":             args.ema,
        "diffusion_steps": args.diffusion_steps,
        "stratified":      args.stratified_timesteps,
        "cfg_weight":      list(args.cfg_weight),
        "reward_eta":      args.reward_eta[0],
        "reward_etas":     list(args.reward_eta),
        "reward_lambdas":  list(args.reward_lambda) if args.reward_lambda else None,
        "guidance_clip":   args.guidance_clip,
        "normalize_guidance": args.normalize_guidance,
        "endpoint_guidance":  args.endpoint_guidance,
        "seed":            args.seed,
        "sample_seed":     args.sample_seed,
        "oracle":          str(oracle_path) if oracle_path else None,
    }
