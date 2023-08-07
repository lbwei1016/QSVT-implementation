import getopt, sys

# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

# Options
options = "hn:a"

# Long options
long_options = ["help", "num_of_qubits_for_matrix=", "AA"]

try:
	# Parsing argument
	arguments, values = getopt.getopt(argumentList, options, long_options)
	
	# checking each argument
	for currentArgument, currentValue in arguments:

		if currentArgument in ("-h", "--help"):
			help_msg = """
                -h: show help
                -n <number of qubits for matrix>: specify matrix size
                -a: Use AA
	        """
			print(help_msg)
			
		elif currentArgument in ("-n", "--num_of_qubits_for_matrix"):
			print(f'n = {currentValue}')
			
		elif currentArgument in ("-a", "--AA"):
			print('AA is on')
			
except getopt.error as err:
	# output error, and return with an error code
	print (str(err))
