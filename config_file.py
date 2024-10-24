learning_params = dict(
    ##########																	#Default Values
    # Irreversible processes
	##########
	annihilation_rule 				= False,									#False
	maximum_terms					= None,										#None

	##########
	# Variance
	##########
	initial_variance 				= 0.3,										#0.3
	variance_update_frequency		= None,										#None
	variance_decrease				= 2,										#2.0
	variance_increase				= 1.5,										#1.5
    
	##########
	# Operator-Space Rates
	##########
    complexity						= 2,										#2.0
	lindbladian_exponential_rate 	= 5,										#5.0
	hamiltonian_exponential_rate	= 2,										#2.0
    
	##########
	# Optimised Learning Parameters
	##########
	PARAMETER_chance		 		= 0.25,										#0.25,
    
    ADDINGH_chance 					= 0.05, 									#0.05
    REMOVINGH_chance				= 0.05,										#.05,
    SWAPPINGH_chance 				= 0.2,										#.1,
    
	ADDINGL_SINGLE_chance			= 0.05,										#0.05
    REMOVINGL_SINGLE_chance 		= 0.05,										#.05,
    SWAPPINGL_SINGLE_chance			= 0.1,										#.1,
    
	ADDINGL_PAIR_chance				= 0.01,										#.01,
    REMOVINGL_PAIR_chance			= 0.01,										#.01,
    SWAPPINGL_PAIR_chance			= 0.02,										#.02,
    
    BASIS_chance 					= 0.02,										#.02,

	parameter_prior_shape 			= 0.03,										#0.03
	parameter_prior_scale			= 20,										#20
    
	parameter_proposal_shape 		= 2,										#2.0
	parameter_proposal_scale		= 0.3,										#0.3
    
	g2_lifetime_percentile			=20,										#20
	##########
	# Easy Access
	##########
	immutable_processes				= ['bkg'],
    immutable_rates					= [None],
	project_name 					= 'Full_Model_Learning_With_5_perc_lifetime_weighting',
	chain_steps						= 10,
)
	