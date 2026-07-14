import jax

jax.config.update("jax_enable_x64", True)

from .contamination import (
    apply_contamination,
    invert_contamination,
    compute_two_point_correction,
    pack_params,
    unpack_params,
    n_free_params,
)
from .likelihood import make_log_likelihood
from .covariance import (
    LowRankPrecision,
    build_lowrank_precision,
    mock_sandwich_covariance,
    build_harmonic_precision,
)
from .maps import (
    systematic_power_spectrum,
    generate_systematic_map,
    generate_systematic_maps,
    load_real_template,
    load_real_templates,
    pixelize_catalog,
    compute_overdensity,
    assign_template_values,
)
from .inference import make_log_prob, run_mcmc, run_additive_analytic, get_mle_params, get_param_variance_from_chain, get_param_covariance_from_chain
from .nuts import run_nuts, build_logdensity, default_n_chains
from .correction import (
    debias_params,
    rotate_templates,
    transform_params_from_rotated,
    correct_two_point_function,
    correct_power_spectrum_harmonic,
)
from .model_selection import (
    likelihood_ratio_test,
    lrt_null_distribution,
    greedy_forward_select,
    GreedyForwardSelectionResult,
    ForwardSelectionRound,
    SnrPreselectionResult,
    snr_preselect,
)
from .bootstrap import block_bootstrap_variance, jackknife_covariance
from .power_spectrum import (
    measure_pseudo_cl,
    subtract_template_cl,
    harmonic_bias,
    mode_projection_bias,
)
from .regression import (
    elasticnet_contamination_fit,
    iterative_systematics_decontamination,
    method_comparison,
    run_decontamination,
)
from .diagnostics import (
    null_test_cross_correlations,
    snr_template_ranking,
    footprint_mask_diagnostics,
    isd_template_significance,
)
from .mocks import (
    generate_lognormal_field,
    make_galactic_mask,
    make_mock_catalog,
    make_mock_suite,
    MockCatalog,
)
from .utils import (
    compute_covariance_matrix,
    compute_amplitude_bias,
    measure_two_point_function,
    measure_cross_two_point_function,
    measure_two_point_function_corrfunc,
    measure_kk_correlation_treecorr,
    measure_kk_correlation_corrfunc,
)
from .plotting import (
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
)
from .glass_mocks import (
    measure_nz,
    generate_glass_fullsky_mock,
    generate_glass_delta_map,
    sample_positions_from_delta,
)
from .simulation import (
    ContaminationConfig,
    LEVELS,
    DEFAULT_MAP_NAMES,
    DEFAULT_METHODS,
    make_contamination_grid,
    load_uchuu_mock,
    load_systematic_maps,
    apply_footprint_mask,
    inject_systematics,
    save_simulation_catalog,
    load_simulation_catalog,
    run_wtheta_recovery,
)

__all__ = [
    # contamination
    "apply_contamination",
    "invert_contamination",
    "compute_two_point_correction",
    "pack_params",
    "unpack_params",
    "n_free_params",
    # likelihood
    "make_log_likelihood",
    # covariance
    "LowRankPrecision",
    "build_lowrank_precision",
    "mock_sandwich_covariance",
    "build_harmonic_precision",
    # maps
    "systematic_power_spectrum",
    "generate_systematic_map",
    "generate_systematic_maps",
    "load_real_template",
    "load_real_templates",
    "pixelize_catalog",
    "compute_overdensity",
    "assign_template_values",
    # inference
    "make_log_prob",
    "run_mcmc",
    "run_additive_analytic",
    "get_mle_params",
    "get_param_variance_from_chain",
    "get_param_covariance_from_chain",
    # nuts
    "run_nuts",
    "build_logdensity",
    "default_n_chains",
    # correction
    "debias_params",
    "rotate_templates",
    "transform_params_from_rotated",
    "correct_two_point_function",
    "correct_power_spectrum_harmonic",
    # model_selection
    "likelihood_ratio_test",
    "lrt_null_distribution",
    "greedy_forward_select",
    "GreedyForwardSelectionResult",
    "ForwardSelectionRound",
    "SnrPreselectionResult",
    "snr_preselect",
    # bootstrap
    "block_bootstrap_variance",
    "jackknife_covariance",
    # utils
    "compute_covariance_matrix",
    "compute_amplitude_bias",
    "measure_two_point_function",
    "measure_cross_two_point_function",
    "measure_two_point_function_corrfunc",
    "measure_kk_correlation_treecorr",
    "measure_kk_correlation_corrfunc",
    # power_spectrum
    "measure_pseudo_cl",
    "subtract_template_cl",
    "harmonic_bias",
    "mode_projection_bias",
    # regression
    "elasticnet_contamination_fit",
    "iterative_systematics_decontamination",
    "method_comparison",
    "run_decontamination",
    # diagnostics
    "null_test_cross_correlations",
    "snr_template_ranking",
    "footprint_mask_diagnostics",
    "isd_template_significance",
    # mocks
    "generate_lognormal_field",
    "make_galactic_mask",
    "make_mock_catalog",
    "make_mock_suite",
    "MockCatalog",
    # plotting
    "METHOD_COLORS",
    "METHOD_LINESTYLES",
    "METHOD_MARKERS",
    "METHOD_LABELS",
    "METHOD_ORDER",
    # glass_mocks
    "measure_nz",
    "generate_glass_fullsky_mock",
    "generate_glass_delta_map",
    "sample_positions_from_delta",
    # simulation
    "ContaminationConfig",
    "LEVELS",
    "DEFAULT_MAP_NAMES",
    "DEFAULT_METHODS",
    "make_contamination_grid",
    "load_uchuu_mock",
    "load_systematic_maps",
    "apply_footprint_mask",
    "inject_systematics",
    "save_simulation_catalog",
    "load_simulation_catalog",
    "run_wtheta_recovery",
]
