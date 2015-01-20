$AUTHOR = 'Coleman'

# implement recurrent neural network with sigmoid, hyperbolic, step, sign, threshold, and squash function
# mark a weight as constant
# cross entropy revision error propagation, and cross entropy derivative error propagation
# cross entropy - mean sum error
# derivative - mean sum squared error
# MSE cost-
#     weight update --- input * output_delta
#     neuron error --- sum(output_delta * output_weight) * sigma_prime
#     output error --- sigma_prime * (target - actual)
#
# CE cost-
#     weight update --- input * output_delta
#     neuron error --- sum(output_delta * output_weight) / num_output
#     output error --- target - actual
