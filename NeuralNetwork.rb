$AUTHOR = 'Coleman'

module ANN
  include Math
  class Network
    def initialize(generations, weights, learning:, momentum: nil, decay: nil, elimination: nil, threshold: nil, minimum:)
      @epoch = 0
      @generations = []
      @bias = Neuron.new self

      @learning = learning
      @momentum = momentum
      @decay = decay
      @elimination = elimination
      @threshold = threshold
      @minimum = minimum

      @weights = weights
      @weightIndex = -1
      # weights are used in assigning biases to node because a bias is the weight of an edge connecting the node to the bias node

      @generations << newIOGeneration(generations[0])
      generations[1...generations.length - 1].each do |data|
        @generations << newGeneration(data)
      end
      @generations << newIOGeneration(generations[-1])

    end
    attr_reader :learning, :momentum, :decay, :elimination, :threshold, :minimum

    def run


    end

    def propagate


    end

    def backpropagate


    end

    def completeConnection(generation, otherGeneration)
      generation.each do |neuron|
        otherGeneration.each do |otherNeuron|
          Synapse.new neuron, otherNeuron, @weights[@weightIndex += 1], self

        end

      end

    end

    private
      def newGeneration(data)
        newGeneration = []
        data.times do
          newGeneration << Neuron.new(self)
          Synapse.new @bias, newGeneration[-1], @weights[@weightIndex += 1], self

        end
        newGeneration

      end

      def newIOGeneration(data)
        newGeneration = []
        data.times do
          newGeneration << Neuron.new(self)

        end
        newGeneration

      end

  end

  class Neuron
    def initialize(net)
      @net = net
      @dendrites = []
      @terminals = []

      @output = 1
      @error = 0
      @prevError = 0

    end
    attr_reader :dendrites, :terminals, :output, :error, :prevError


    def activate
      sum = 0
      @dendrites.each do |synapse|
        sum += synapse.input * synapse.weight

      end

      @output = sigmoid sum

      @terminals.each do |synapse|
        synapse.transfer

      end

    end

    def learn
      sum = 0
      @terminals.each do |synapse|
        sum += synapse.dendrite.error * synapse.weight

      end

      @prevError = @error

      @error = derivative(@output) * sum
      @error = 0 if @error.abs <= @net.minimum

      terminals.each do |synapse|
        synapse.update

      end

    end

    def sigmoid(value)
      1 / (1 + Math::E ** -value)

    end

    def threshold
      output = 1 / (1 + Math::E ** -value)
      unless @net.threshold.nil?
        if output < @net.threshold
          0
        else
          1
        end

      end

    end

    def derivative(value)
      value * (1 - value)

    end

  end

  class Synapse
    def initialize(terminal, dendrite, weight, net)
      @net = net
      @terminal = terminal
      @dendrite = dendrite
      @weight = weight
      @input = 0

      terminal.terminals << self
      dendrite.dendrites << self

      # @hormone = 0
      # self
      # commented lines to be used in the effect of hormones on artificial neural networks and multi-learning

    end
    attr_reader :input, :weight

    def transfer
      @input = @terminal.output

    end

    def update
      @weight += @net.learning * (@input * @dendrite.error) # simple update equation

    end

  end

  # always use target - actual for error calculation, not actual - target

end