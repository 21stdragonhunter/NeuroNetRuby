$AUTHOR = 'Coleman'

module ANN
  include Math
  class Network
    def printNetwork
      @synapses.each do |synapse|
        synapse.printSynapse

      end

    end

    def initialize(generations, learning:, momentum: nil, decay: nil, elimination: nil, threshold: nil, minimum:)
      @epoch = 0
      @generations = []
      @synapses = []
      @bias = Neuron.new self, "Bias"

      @learning = learning
      @momentum = momentum
      @decay = decay
      @elimination = elimination
      @threshold = threshold
      @minimum = minimum

      @generations << newInputGeneration(generations[0], 0)
      generations[1...generations.length].zip(1...generations.length).each do |data, index|
        @generations << newGeneration(data, index)
      end

    end
    attr_reader :learning, :momentum, :decay, :elimination, :threshold, :minimum, :synapses

    def run(inputs, targets, maxEpoch, pause, minError: Float::INFINITY)
      @generations.each do |generation|
        generation.each do |neuron|
          neuron.initializeDendrites

        end

      end

      printNetwork

      inputsTargets = inputs.zip(targets)

      maxEpoch.times do
        puts "Epoch: #{@epoch}"

        minError = [inputs.length, minError].min
        error = 0

        inputsTargets.shuffle!

        inputsTargets.each do |input, target|
          output = propagate input
          error += backpropagate target
          puts "#{input}: #{output}"

        end

        break if minError <= error

        gets if @epoch % pause == 0
        @epoch += 1

      end

    end

    def completeConnection(generation, otherGeneration)
      @generations[generation].each do |neuron|
        @generations[otherGeneration].each do |otherNeuron|
          Synapse.new neuron, otherNeuron, self

        end

      end

    end

    private
      def propagate(inputs)
        @generations[0].zip(inputs).each do |neuron, input|
          neuron.output = input

        end
        @generations[1...@generations.length].each do |generation|
          generation.each do |neuron|
            neuron.activate(unless @threshold.nil?
                              :threshold
                            else
                              nil
                            end)

          end

        end

        outputs = []
        @generations[-1].each do |neuron|
          outputs << neuron.output

        end
        outputs

      end

      def backpropagate(targets)
        error = 0
        @generations[-1].zip(targets).each do |neuron, target|
          neuron.error = target - neuron.output
          neuron.error = 0 if neuron.error.abs <= @minimum
          error += neuron.error

        end

        return 1 if error == 0

        @generations[0...@generations.length - 1].reverse.each do |generation|
          generation.each do |neuron|
            neuron.learn

          end

        end

        @synapses.each do |synapse|
          synapse.update(unless @momentum.nil?
                             :momentum
                           else
                             unless @decay.nil? || @elimination.nil?
                               :elimination
                             else
                               nil
                             end
                           end)

        end

        return 0

      end

      def newGeneration(data, index)
        newGeneration = []
        (0...data).each do |neuronIndex|
          newGeneration << Neuron.new(self, [index, neuronIndex])
          Synapse.new @bias, newGeneration[-1], self

        end
        newGeneration

      end

      def newInputGeneration(data, index)
        newGeneration = []
        (0...data).each do |neuronIndex|
          newGeneration << Neuron.new(self, [index, neuronIndex])

        end
        newGeneration

      end

  end

  class Neuron
    def initialize(net, coordinates)
      @coordinates = coordinates
      @net = net
      @dendrites = []
      @terminals = []

      @output = 1
      @error = 0
      @prevError = 0

    end
    def initializeDendrites
      weights = []
      @dendrites.length.times do
        weights << Random.rand(-1.0...1.0)
      end

      @dendrites.zip(weights).each do |synapse, weight|
        synapse.weight = weight

      end

    end
    attr_accessor :output, :error
    attr_reader :dendrites, :terminals, :prevError, :coordinates


    def activate(type)
      sum = 0
      @dendrites.each do |synapse|
        sum += synapse.terminal.output * synapse.weight

      end

      @output = (if type.nil?
                   hyperbolic sum
                 else
                   threshold sum
                 end)

    end

    def learn
      sum = 0
      @terminals.each do |synapse|
        sum += synapse.dendrite.error * synapse.weight

      end

      @prevError = @error

      @error = derivativeHyperbolic(@output) * sum
      @error = 0 if @error.abs < @net.minimum

    end

    def sigmoid(value)
      1 / (1 + Math::E ** -value)

    end

    def hyperbolic(value)
      (Math::E ** (2 * value) - 1)/(Math::E ** (2 * value) + 1)

    end

    def threshold(value)
      output = 1 / (1 + Math::E ** -value)
      unless @net.threshold.nil?
        if output < @net.threshold
          0
        else
          1
        end

      end

    end

    def derivativeSigmoid(value)
      value * (1 - value)

    end

    def derivativeHyperbolic(value)
      1 - value ** 2

    end

  end

  class Synapse
    def printSynapse
      puts "From: #{@terminal.coordinates} To: #{@dendrite.coordinates} Weight: #{@weight}"

    end

    def initialize(terminal, dendrite, net)
      @net = net
      @terminal = terminal
      @dendrite = dendrite
      @weight = 0

      terminal.terminals << self
      dendrite.dendrites << self
      net.synapses << self

      # @hormone = 0
      # self
      # commented lines to be used in the effect of hormones on artificial neural networks and multi-learning

    end
    attr_accessor :weight#, :hormone
    attr_reader :terminal, :dendrite

    def update(type)
      if type.nil?
        @weight += @net.learning * (@terminal.output * @dendrite.error)
      elsif type == :momentum
        @weight += @net.learning * (@terminal.output * (@dendrite.error + (@net.momentum * @dendrite.prevError)))
      else
        @weight += @net.learning * (@terminal.output * @dendrite.error)
        # stub, to be implemented with lamdbda((omega**2 / epsilon**2) / (1 + (omega**2 / epsilon**2)))
      end

    end

  end

end

include ANN

test = Network.new [2, 2, 1], learning: 0.05, minimum: 0.001, momentum: 0.01
test.completeConnection 0, 1
test.completeConnection 1, 2

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

targets = [
    [0],
    [1],
    [1],
    [0]
]

test.run inputs, targets, 2_000_000, 1_000_000

test.printNetwork