  -pc_type <now gamg : formerly (null)>: Preconditioner (one of) nn hypre hmg tfs composite ksp lu icc patch bjacobi eisenstat smg deflation pfmg vpbjacobi redistribute sor mg pbjacobi syspfmg cholesky python mat qr svd fieldsplit bddc mpi kaczmarz jacobi telescope redundant cp shell galerkin ilu exotic gasm gamg none lmvm asm lsc (PCSetType)
  -pc_gamg_type <now agg : formerly agg>: Type of AMG method (only 'agg' supported and useful) (one of) classical geo agg (PCGAMGSetType)
  -pc_gamg_repartition: <now FALSE : formerly FALSE> Repartion coarse grids (PCGAMGSetRepartition)
  -pc_gamg_use_sa_esteig: <now TRUE : formerly TRUE> Use eigen estimate from smoothed aggregation for smoother (PCGAMGSetUseSAEstEig)
  -pc_gamg_recompute_esteig: <now TRUE : formerly TRUE> Set flag to recompute eigen estimates for Chebyshev when matrix changes (PCGAMGSetRecomputeEstEig)
  -pc_gamg_reuse_interpolation: <now TRUE : formerly TRUE> Reuse prolongation operator (PCGAMGReuseInterpolation)
  -pc_gamg_asm_use_agg: <now FALSE : formerly FALSE> Use aggregation aggregates for ASM smoother (PCGAMGASMSetUseAggs)
  -pc_gamg_parallel_coarse_grid_solver: <now FALSE : formerly FALSE> Use parallel coarse grid solver (otherwise put last grid on one process) (PCGAMGSetParallelCoarseGridSolve)
  -pc_gamg_cpu_pin_coarse_grids: <now FALSE : formerly FALSE> Pin coarse grids to the CPU (PCGAMGSetCpuPinCoarseGrids)
  -pc_gamg_coarse_grid_layout_type: <now spread : formerly spread> compact: place reduced grids on processes in natural order; spread: distribute to whole machine for more memory bandwidth (choose one of) compact spread (PCGAMGSetCoarseGridLayoutType)
  -pc_gamg_process_eq_limit: <now 50 : formerly 50>: Limit (goal) on number of equations per process on coarse grids (PCGAMGSetProcEqLim)
  -pc_gamg_coarse_eq_limit: <now 50 : formerly 50>: Limit on number of equations for the coarse grid (PCGAMGSetCoarseEqLim)
  -pc_gamg_threshold_scale: <now 1. : formerly 1.>: Scaling of threshold for each level not specified (PCGAMGSetThresholdScale)
  -pc_gamg_threshold: <-1.>: Relative threshold to use for dropping edges in aggregation graph (PCGAMGSetThreshold)
  -pc_gamg_rank_reduction_factors: <0>: Manual schedule of coarse grid reduction factors that overrides internal heuristics (0 for first reduction puts one process/device) (PCGAMGSetRankReductionFactors)
  -pc_gamg_eigenvalues: <0.>: extreme eigenvalues for smoothed aggregation (PCGAMGSetEigenvalues)
  -pc_gamg_agg_nsmooths: <now 1 : formerly 1>: smoothing steps for smoothed aggregation, usually 1 (PCGAMGSetNSmooths)
  -pc_gamg_aggressive_coarsening: <now 1 : formerly 1>: Number of aggressive coarsening (MIS-2) levels from finest (PCGAMGSetAggressiveLevels)
  -pc_gamg_square_graph: <now 0 : formerly 0>: Number of aggressive coarsening (MIS-2) levels from finest (deprecated alias for -pc_gamg_aggressive_coarsening) (PCGAMGSetAggressiveLevels)
  -pc_gamg_aggressive_square_graph: <now TRUE : formerly TRUE> Use square graph (A'A) or MIS-k (k=2) for aggressive coarsening (PCGAMGSetAggressiveSquareGraph)
  -pc_gamg_mis_k_minimum_degree_ordering: <now FALSE : formerly FALSE> Use minimum degree ordering for greedy MIS (PCGAMGMISkSetMinDegreeOrdering)
  -pc_gamg_low_memory_threshold_filter: <now FALSE : formerly FALSE> Use the (built-in) low memory graph/matrix filter (PCGAMGSetLowMemoryFilter)
  -pc_gamg_aggressive_mis_k: <now 2 : formerly 2>: Number of levels of multigrid to use. (PCGAMGMISkSetAggressive)



  -pc_type <now mg : formerly (null)>: Preconditioner (one of) nn hypre hmg tfs composite ksp lu icc patch bjacobi eisenstat smg deflation pfmg vpbjacobi redistribute sor mg pbjacobi syspfmg cholesky python mat qr svd fieldsplit bddc mpi kaczmarz jacobi telescope redundant cp shell galerkin ilu exotic gasm gamg none lmvm asm lsc (PCSetType)
  -pc_mg_levels: <now 1 : formerly 1>: Number of Levels (PCMGSetLevels)
  -pc_mg_cycle_type: <now v : formerly v> V cycle or for W-cycle (choose one of) invalid v w (PCMGSetCycleType)
  -pc_mg_galerkin: <now none : formerly none> Use Galerkin process to compute coarser operators (choose one of) both pmat mat none external (PCMGSetGalerkin)
  -pc_mg_adapt_interp_coarse_space: <now none : formerly none> Type of adaptive coarse space: none, polynomial, harmonic, eigenvector, generalized_eigenvector, gdsw (choose one of) none polynomial harmonic eigenvector generalized_eigenvector gdsw (PCMGSetAdaptCoarseSpaceType)
  -pc_mg_adapt_interp_n: <now -1 : formerly -1>: Size of the coarse space for adaptive interpolation (PCMGSetCoarseSpace)
  -pc_mg_mesp_monitor: <now FALSE : formerly FALSE> Monitor the multilevel eigensolver (PCMGSetAdaptInterpolation)
  -pc_mg_adapt_cr: <now FALSE : formerly FALSE> Monitor coarse space quality using Compatible Relaxation (CR) (PCMGSetAdaptCR)
  -pc_mg_distinct_smoothup: <now FALSE : formerly FALSE> Create separate smoothup KSP and append the prefix _up (PCMGSetDistinctSmoothUp)
  -pc_mg_type: <now MULTIPLICATIVE : formerly MULTIPLICATIVE> Multigrid type (choose one of) MULTIPLICATIVE ADDITIVE FULL KASKADE (PCMGSetType)
  -pc_mg_multiplicative_cycles: <now 1 : formerly 1>: Number of cycles for each preconditioner step (PCMGMultiplicativeSetCycles)
  -pc_mg_log: <now FALSE : formerly FALSE> Log times for each multigrid level (None)




  Options for all PETSc programs:
 -version: prints PETSc version
 -help intro: prints example description and PETSc version, and exits
 -help: prints example description, PETSc version, and available options for used routines
 -on_error_abort: cause an abort when an error is detected. Useful 
        only when run in the debugger
 -on_error_attach_debugger [gdb,dbx,xxgdb,ups,noxterm]
       start the debugger in new xterm
       unless noxterm is given
 -start_in_debugger [gdb,dbx,xxgdb,ups,noxterm]
       start all processes in the debugger
 -on_error_emacs <machinename>
    emacs jumps to error file
 -debugger_ranks [n1,n2,..] Ranks to start in debugger
 -debugger_pause [m] : delay (in seconds) to attach debugger
 -stop_for_debugger : prints message on how to attach debugger manually
                      waits the delay for you to attach
 -display display: Location where X window graphics and debuggers are displayed
 -no_signal_handler: do not trap error signals
 -mpi_return_on_error: MPI returns error code, rather than abort on internal error
 -fp_trap: stop on floating point exceptions
           note on IBM RS6000 this slows run greatly
 -malloc_dump <optional filename>: dump list of unfreed memory at conclusion
 -on_error_malloc_dump <optional filename>: dump list of unfreed memory on memory error
 -malloc_view <optional filename>: keeps log of all memory allocations, displays in PetscFinalize()
 -malloc_debug <true or false>: enables or disables extended checking for memory corruption
 -options_view: dump list of options inputted
 -options_left: dump list of unused options
 -options_left no: don't dump list of unused options
 -tmp tmpdir: alternative /tmp directory
 -shared_tmp: tmp directory is shared by all processors
 -not_shared_tmp: each processor has separate tmp directory
 -memory_view: print memory usage at end of run
 -get_total_flops: total flops over all processors
 -log_view [:filename:[format]]: logging objects and events
 -log_trace [filename]: prints trace of all PETSc calls
 -log_exclude <list,of,classnames>: exclude given classes from logging
 -info [filename][:[~]<list,of,classnames>[:[~]self]]: print verbose information
 -options_file <file>: reads options from file
 -options_monitor: monitor options to standard output, including that set previously e.g. in option files
 -options_monitor_cancel: cancels all hardwired option monitors
 -petsc_sleep n: sleeps n seconds before running program
----------------------------------------
PetscDevice Options:
  -device_enable: <now lazy : formerly lazy> How (or whether) to initialize PetscDevices (choose one of) none lazy eager (PetscDeviceInitialize())
  -default_device_type: <now host : formerly host> Set the PetscDeviceType returned by PETSC_DEVICE_DEFAULT() (choose one of) host cuda hip sycl (PetscDeviceSetDefaultDeviceType())
  -device_select: <now -1 : formerly -1>: Which device to use. Pass (-1) to have PETSc decide or (given they exist) [0-128) for a specific device (PetscDeviceCreate())
  -device_view: <now FALSE : formerly FALSE> Display device information and assignments (forces eager initialization) (PetscDeviceView())
----------------------------------------
PetscDevice host Options:
  -device_view_host: <now FALSE : formerly FALSE> Display device information and assignments (forces eager initialization) (PetscDeviceView())
----------------------------------------
Root PetscDeviceContext Options:
  -root_device_context_device_type: <now host : formerly host> Underlying PetscDevice (choose one of) host cuda hip sycl (PetscDeviceContextSetDevice)
  -root_device_context_stream_type: <now global_blocking : formerly global_blocking> PetscDeviceContext PetscStreamType (choose one of) global_blocking default_blocking global_nonblocking (PetscDeviceContextSetStreamType)
----------------------------------------
BLAS options:
  -blas_view: Display number of threads to use for BLAS operations (None)
----------------------------------------
Preconditioner (PC) options:
  -pc_type <now mg : formerly (null)>: Preconditioner (one of) nn hypre hmg tfs composite ksp lu icc patch bjacobi eisenstat smg deflation pfmg vpbjacobi redistribute sor mg pbjacobi syspfmg cholesky python mat qr svd fieldsplit bddc mpi kaczmarz jacobi telescope redundant cp shell galerkin ilu exotic gasm gamg none lmvm asm lsc (PCSetType)
  -pc_use_amat: <now TRUE : formerly TRUE> use Amat (instead of Pmat) to define preconditioner in nested inner solves (PCSetUseAmat)
  Multigrid options
  -pc_mg_levels: <now 1 : formerly 1>: Number of Levels (PCMGSetLevels)
  -pc_mg_cycle_type: <now v : formerly v> V cycle or for W-cycle (choose one of) invalid v w (PCMGSetCycleType)
  -pc_mg_galerkin: <now none : formerly none> Use Galerkin process to compute coarser operators (choose one of) both pmat mat none external (PCMGSetGalerkin)
  -pc_mg_adapt_interp_coarse_space: <now none : formerly none> Type of adaptive coarse space: none, polynomial, harmonic, eigenvector, generalized_eigenvector, gdsw (choose one of) none polynomial harmonic eigenvector generalized_eigenvector gdsw (PCMGSetAdaptCoarseSpaceType)
  -pc_mg_adapt_interp_n: <now -1 : formerly -1>: Size of the coarse space for adaptive interpolation (PCMGSetCoarseSpace)
  -pc_mg_mesp_monitor: <now FALSE : formerly FALSE> Monitor the multilevel eigensolver (PCMGSetAdaptInterpolation)
  -pc_mg_adapt_cr: <now FALSE : formerly FALSE> Monitor coarse space quality using Compatible Relaxation (CR) (PCMGSetAdaptCR)
  -pc_mg_distinct_smoothup: <now FALSE : formerly FALSE> Create separate smoothup KSP and append the prefix _up (PCMGSetDistinctSmoothUp)
  -pc_mg_type: <now MULTIPLICATIVE : formerly MULTIPLICATIVE> Multigrid type (choose one of) MULTIPLICATIVE ADDITIVE FULL KASKADE (PCMGSetType)
  -pc_mg_multiplicative_cycles: <now 1 : formerly 1>: Number of cycles for each preconditioner step (PCMGMultiplicativeSetCycles)
  -pc_mg_log: <now FALSE : formerly FALSE> Log times for each multigrid level (None)
----------------------------------------
Krylov Method (KSP) options:
  -ksp_type <now gmres : formerly gmres>: Krylov method (one of) fetidp pipefgmres stcg tsirm tcqmr groppcg nash fcg symmlq lcd minres cgs preonly lgmres pipecgrr fbcgs pipeprcg pipecg ibcgs fgmres qcg gcr cgne pipefcg pipecr pipebcgs bcgsl pipecg2 pipelcg gltr cg tfqmr pgmres lsqr pipegcr bicg cgls bcgs cr python dgmres none qmrcgs gmres richardson chebyshev fbcgsr (KSPSetType)
  -ksp_monitor_cancel: <now FALSE : formerly FALSE> Remove any hardwired monitor routines (KSPMonitorCancel)
----------------------------------------
Viewer (-ksp_monitor) options:
  -ksp_monitor ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_monitor_short) options:
  -ksp_monitor_short ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor_short binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor_short draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor_short socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor_short saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-all_ksp_monitor) options:
  -all_ksp_monitor ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -all_ksp_monitor binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -all_ksp_monitor draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -all_ksp_monitor socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -all_ksp_monitor saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_monitor_range) options:
  -ksp_monitor_range ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor_range binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor_range draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor_range socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor_range saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_monitor_true_residual) options:
  -ksp_monitor_true_residual ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor_true_residual binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor_true_residual draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor_true_residual socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor_true_residual saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_monitor_max) options:
  -ksp_monitor_max ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor_max binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor_max draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor_max socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor_max saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_monitor_solution) options:
  -ksp_monitor_solution ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor_solution binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor_solution draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor_solution socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor_solution saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_monitor_singular_value) options:
  -ksp_monitor_singular_value ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor_singular_value binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor_singular_value draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor_singular_value socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor_singular_value saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_monitor_error) options:
  -ksp_monitor_error ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_monitor_error binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_monitor_error draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_monitor_error socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_monitor_error saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
  -ksp_monitor_pause_final: <now FALSE : formerly FALSE> Pauses all draw monitors at the final iterate (KSPMonitorPauseFinal_Internal)
  -ksp_initial_guess_nonzero: <now FALSE : formerly FALSE> Use the contents of the solution vector for initial guess (KSPSetInitialNonzero)
  -ksp_max_it: <now 10000 : formerly 10000>: Maximum number of iterations (KSPSetTolerances)
  -ksp_min_it: <now 0 : formerly 0>: Minimum number of iterations (KSPSetMinimumIterations)
  -ksp_rtol: <now 1e-05 : formerly 1e-05>: Relative decrease in residual norm (KSPSetTolerances)
  -ksp_atol: <now 1e-50 : formerly 1e-50>: Absolute value of residual norm (KSPSetTolerances)
  -ksp_divtol: <now 10000. : formerly 10000.>: Residual norm increase cause divergence (KSPSetTolerances)
  -ksp_converged_use_initial_residual_norm: <now FALSE : formerly FALSE> Use initial residual norm for computing relative convergence (KSPConvergedDefaultSetUIRNorm)
  -ksp_converged_use_min_initial_residual_norm: <now FALSE : formerly FALSE> Use minimum of initial residual norm and b for computing relative convergence (KSPConvergedDefaultSetUMIRNorm)
  -ksp_converged_maxits: <now FALSE : formerly FALSE> Declare convergence if the maximum number of iterations is reached (KSPConvergedDefaultSetConvergedMaxits)
  -ksp_converged_neg_curve: <now FALSE : formerly FALSE> Declare convergence if negative curvature is detected (KSPConvergedNegativeCurvature)
  -ksp_reuse_preconditioner: <now FALSE : formerly FALSE> Use initial preconditioner and don't ever compute a new one (KSPReusePreconditioner)
  -ksp_knoll: <now FALSE : formerly FALSE> Use preconditioner applied to b for initial guess (KSPSetInitialGuessKnoll)
  -ksp_error_if_not_converged: <now FALSE : formerly FALSE> Generate error if solver does not converge (KSPSetErrorIfNotConverged)
  -ksp_guess_type <now (null) : formerly (null)>: Initial guess in Krylov method (one of) pod fischer ((null))
  -ksp_fischer_guess: <0>: Use Paul Fischer's algorithm or its variants for initial guess (KSPSetUseFischerGuess)
  -ksp_convergence_test: <now default : formerly default> Convergence test (choose one of) default skip lsqr (KSPSetConvergenceTest)
  -ksp_norm_type: <now PRECONDITIONED : formerly PRECONDITIONED> KSP Norm type (choose one of) NONE PRECONDITIONED UNPRECONDITIONED NATURAL (KSPSetNormType)
  -ksp_check_norm_iteration: <now -1 : formerly -1>: First iteration to compute residual norm (KSPSetCheckNormIteration)
  -ksp_lag_norm: <now FALSE : formerly FALSE> Lag the calculation of the residual norm (KSPSetLagNorm)
  -ksp_diagonal_scale: <now FALSE : formerly FALSE> Diagonal scale matrix before building preconditioner (KSPSetDiagonalScale)
  -ksp_diagonal_scale_fix: <now FALSE : formerly FALSE> Fix diagonally scaled matrix after solve (KSPSetDiagonalScaleFix)
  -ksp_constant_null_space: <now FALSE : formerly FALSE> Add constant null space to Krylov solver matrix (MatSetNullSpace)
  -ksp_monitor_python: <now (null) : formerly (null)>: Use Python function (KSPMonitorSet)
  -ksp_monitor_lg_range: <now FALSE : formerly FALSE> Monitor graphically range of preconditioned residual norm (KSPMonitorSet)
----------------------------------------
Viewer (-ksp_view) options:
  -ksp_view ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_pre) options:
  -ksp_view_pre ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_pre binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_pre draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_pre socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_pre saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
  -ksp_converged_reason_view_cancel: <now FALSE : formerly FALSE> Cancel all the converged reason view functions set using KSPConvergedReasonViewSet (KSPConvergedReasonViewCancel)
----------------------------------------
Viewer (-ksp_converged_rate) options:
  -ksp_converged_rate ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_converged_rate binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_converged_rate draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_converged_rate socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_converged_rate saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_mat) options:
  -ksp_view_mat ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_mat binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_mat draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_mat socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_mat saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_pmat) options:
  -ksp_view_pmat ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_pmat binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_pmat draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_pmat socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_pmat saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_rhs) options:
  -ksp_view_rhs ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_rhs binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_rhs draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_rhs socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_rhs saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_solution) options:
  -ksp_view_solution ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_solution binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_solution draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_solution socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_solution saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_mat_explicit) options:
  -ksp_view_mat_explicit ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_mat_explicit binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_mat_explicit draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_mat_explicit socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_mat_explicit saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_eigenvalues) options:
  -ksp_view_eigenvalues ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_eigenvalues binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_eigenvalues draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_eigenvalues socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_eigenvalues saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_singularvalues) options:
  -ksp_view_singularvalues ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_singularvalues binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_singularvalues draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_singularvalues socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_singularvalues saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_eigenvalues_explicit) options:
  -ksp_view_eigenvalues_explicit ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_eigenvalues_explicit binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_eigenvalues_explicit draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_eigenvalues_explicit socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_eigenvalues_explicit saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_final_residual) options:
  -ksp_view_final_residual ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_final_residual binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_final_residual draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_final_residual socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_final_residual saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_preconditioned_operator_explicit) options:
  -ksp_view_preconditioned_operator_explicit ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_preconditioned_operator_explicit binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_preconditioned_operator_explicit draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_preconditioned_operator_explicit socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_preconditioned_operator_explicit saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
----------------------------------------
Viewer (-ksp_view_diagonal_scale) options:
  -ksp_view_diagonal_scale ascii[:[filename][:[format][:append]]]: Prints object to stdout or ASCII file (PetscOptionsGetViewer)
  -ksp_view_diagonal_scale binary[:[filename][:[format][:append]]]: Saves object to a binary file (PetscOptionsGetViewer)
  -ksp_view_diagonal_scale draw[:[drawtype][:filename|format]] Draws object (PetscOptionsGetViewer)
  -ksp_view_diagonal_scale socket[:port]: Pushes object to a Unix socket (PetscOptionsGetViewer)
  -ksp_view_diagonal_scale saws[:communicatorname]: Publishes object to SAWs (PetscOptionsGetViewer)
  -ksp_plot_eigenvalues: [deprecated since PETSc 3.9; use -ksp_view_eigenvalues draw] (KSPView)
  -ksp_plot_eigencontours: [deprecated since PETSc 3.9; use -ksp_view_eigenvalues draw::draw_contour] (KSPView)
  -ksp_plot_eigenvalues_explicitly: [deprecated since PETSc 3.9; use -ksp_view_eigenvalues_explicit draw] (KSPView)
  -ksp_pc_side: <now LEFT : formerly LEFT> KSP preconditioner side (choose one of) LEFT RIGHT SYMMETRIC (KSPSetPCSide)
  -ksp_matsolve_batch_size: <now -1 : formerly -1>: Maximum number of columns treated simultaneously (KSPSetMatSolveBatchSize)
  -ksp_use_explicittranspose: <now FALSE : formerly FALSE> Explicitly transpose the system in KSPSolveTranspose (KSPSetUseExplicitTranspose)
  KSP GMRES Options
  -ksp_gmres_restart: <now 30 : formerly 30>: Number of Krylov search directions (KSPGMRESSetRestart)
  -ksp_gmres_haptol: <now 1e-30 : formerly 1e-30>: Tolerance for exact convergence (happy ending) (KSPGMRESSetHapTol)
  -ksp_gmres_breakdown_tolerance: <now 0.1 : formerly 0.1>: Divergence breakdown tolerance during GMRES restart (KSPGMRESSetBreakdownTolerance)
  -ksp_gmres_preallocate: <now FALSE : formerly FALSE> Preallocate Krylov vectors (KSPGMRESSetPreAllocateVectors)
  Pick at most one of -------------
    -ksp_gmres_classicalgramschmidt: Classical (unmodified) Gram-Schmidt (fast) (KSPGMRESSetOrthogonalization)
    -ksp_gmres_modifiedgramschmidt: Modified Gram-Schmidt (slow,more stable) (KSPGMRESSetOrthogonalization)
  -ksp_gmres_cgs_refinement_type: <now REFINE_NEVER : formerly REFINE_NEVER> Type of iterative refinement for classical (unmodified) Gram-Schmidt (choose one of) REFINE_NEVER REFINE_IFNEEDED REFINE_ALWAYS (KSPGMRESSetCGSRefinementType)
  -ksp_gmres_krylov_monitor: <now FALSE : formerly FALSE> Plot the Krylov directions (KSPMonitorSet)


    -pc_hypre_type: <now boomeramg : formerly boomeramg> HYPRE preconditioner type (choose one of) euclid pilut parasails boomeramg ams ads (PCHYPRESetType)
  -pc_hypre_boomeramg_cycle_type: <now V : formerly V> Cycle type (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels: <now 25 : formerly 25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter: <now 1 : formerly 1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol: <now 0. : formerly 0.>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_numfunctions: <now 1 : formerly 1>: Number of functions (HYPRE_BoomerAMGSetNumFunctions)
  -pc_hypre_boomeramg_truncfactor: <now 0. : formerly 0.>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max: <now 0 : formerly 0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl: <now 0 : formerly 0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths: <now 1 : formerly 1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold: <now 0.25 : formerly 0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum: <now 0.9 : formerly 0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all: <now 1 : formerly 1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_nodal_coarsen: <now 0 : formerly 0>: Use a nodal based coarsening 1-6 (HYPRE_BoomerAMGSetNodal)
  -pc_hypre_boomeramg_nodal_coarsen_diag: <now 0 : formerly 0>: Diagonal in strength matrix for nodal based coarsening 0-2 (HYPRE_BoomerAMGSetNodalDiag)
  -pc_hypre_boomeramg_vec_interp_variant: <now 0 : formerly 0>: Variant of algorithm 1-3 (HYPRE_BoomerAMGSetInterpVecVariant)
  -pc_hypre_boomeramg_vec_interp_qmax: <now 0 : formerly 0>: Max elements per row for each Q (HYPRE_BoomerAMGSetInterpVecQMax)
  -pc_hypre_boomeramg_vec_interp_smooth: <now FALSE : formerly FALSE> Whether to smooth the interpolation vectors (HYPRE_BoomerAMGSetSmoothInterpVectors)
  -pc_hypre_boomeramg_interp_refine: <now 0 : formerly 0>: Preprocess the interpolation matrix through iterative weight refinement (HYPRE_BoomerAMGSetInterpRefine)
  -pc_hypre_boomeramg_grid_sweeps_down: <now 1 : formerly 1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up: <now 1 : formerly 1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse: <now 1 : formerly 1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_smooth_type: <now Schwarz-smoothers : formerly Schwarz-smoothers> Enable more complex smoothers (choose one of) Schwarz-smoothers Pilut ParaSails Euclid (None)
  -pc_hypre_boomeramg_smooth_num_levels: <now 25 : formerly 25>: Number of levels on which more complex smoothers are used (None)
  -pc_hypre_boomeramg_eu_level: <now 0 : formerly 0>: Number of levels for ILU(k) in Euclid smoother (None)
  -pc_hypre_boomeramg_eu_droptolerance: <now 0. : formerly 0.>: Drop tolerance for ILU(k) in Euclid smoother (None)
  -pc_hypre_boomeramg_eu_bj: <now FALSE : formerly FALSE> Use Block Jacobi for ILU in Euclid smoother? (None)
  -pc_hypre_boomeramg_relax_type_all: <now symmetric-SOR/Jacobi : formerly symmetric-SOR/Jacobi> Relax type for the up and down cycles (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination    l1-Gauss-Seidel backward-l1-Gauss-Seidel CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down: <now symmetric-SOR/Jacobi : formerly symmetric-SOR/Jacobi> Relax type for the down cycles (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination    l1-Gauss-Seidel backward-l1-Gauss-Seidel CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up: <now symmetric-SOR/Jacobi : formerly symmetric-SOR/Jacobi> Relax type for the up cycles (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination    l1-Gauss-Seidel backward-l1-Gauss-Seidel CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse: <now Gaussian-elimination : formerly Gaussian-elimination> Relax type on coarse grid (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination    l1-Gauss-Seidel backward-l1-Gauss-Seidel CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all: <now 1. : formerly 1.>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level: <1.>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all: <now 1. : formerly 1.>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level: <1.>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <now FALSE : formerly FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type: <now local : formerly local> Measure type (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type: <now Falgout : formerly Falgout> Coarsen type (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_max_coarse_size: <now 9 : formerly 9>: Maximum size of coarsest grid (None)
  -pc_hypre_boomeramg_min_coarse_size: <now 1 : formerly 1>: Minimum size of coarsest grid (None)
  -pc_hypre_boomeramg_restriction_type: <now 0 : formerly 0>: Type of AIR method (distance 1 or 2, 0 means no AIR) (None)
  -pc_hypre_boomeramg_interp_type: <now classical : formerly classical> Interpolation type (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts block block-wtd FF FF1 ext ad-wts ext-mm ext+i-mm ext+e-mm (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_relaxation: <now FALSE : formerly FALSE> Nodal relaxation via Schwarz (None)
  -pc_hypre_boomeramg_keeptranspose: <now FALSE : formerly FALSE> Avoid transpose matvecs in preconditioner application (None)
  -pc_hypre_boomeramg_parasails_sym: <now nonsymmetric : formerly nonsymmetric> Symmetry of matrix and preconditioner (choose one of) nonsymmetric SPD nonsymmetric,SPD (None)