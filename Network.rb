__author__ = "Coleman"

module NeuralNetwork
  include Math
  class Network
      def initialize(generationData, biases, weights, learning: , momentum: , decay: , minimum: , threshold: nil)
        @epoch = 0
        @generations = []
        @biases = biases
        @weights = weights
        @weightIndex = -1
        @biasIndex = -1

        @learning = learning
        @momentum = momentum
        @decay = decay
        @threshold = threshold
        @minimum = minimum

        @generations << newIOGeneration(generationData[0])

        lastGeneration = generationData[-1]
        generationData = generationData[1...generationData.length - 1]

        generationData.each do |data|
          @generations << newGeneration(data)

        end

        @generations << newIOGeneration(lastGeneration)

        (0...@generations.length - 1).each do |i|
          completeConnection i, i + 1

        end

      end

      def epoch=(epoch)
        @epoch = epoch
      end

      def epoch
        @epoch
      end
      def learning
        @learning
      end
      def momentum
        @momentum
      end
      def decay
        @decay
      end
      def minimum
        @minimum
      end
      def threshold
        @threshold
      end

      def propagate(inputs)
        @generations[0].zip(inputs).each do |neuron, input|
          neuron.output = input
          neuron.terminals.each do |synapse|
            synapse.transfer

          end

        end

        generations = @generations[1...@generations.length]
        generations.each do |generation|
          generation.each do |neuron|
            neuron.activate

          end

        end

        outputs = []
        @generations[-1].each do |neuron|
          outputs << neuron.output

        end

        outputs

      end

      def backpropagate(targets)
        errorSum = 0

        @generations[-1].zip(targets).each do |neuron, target|
          error = target - neuron.output
          error = 0 if error.abs <= @minimum
          neuron.error = error
          errorSum = 1 if error != 0

        end

        return 1 if errorSum == 0

        generations = @generations[0...@generations.length - 1].reverse
        generations.each do |generation|
          generation.each do |neuron|
            neuron.train

          end

        end

        return 0

      end

      def printNetwork
        @generations.zip(0...@generations.length).each do |generation, index|
          print "Generation: "
          puts index
          generation.zip(0...generation.length).each do |neuron, index|
            print "  "
            print index
            print ": "
            neuron.printNeuron

          end

        end

      end

    private
      def newGeneration(size)
        generation = []
        size.times do
          generation << Neuron.new(self, @biases[@biasIndex += 1])

        end
        generation

      end

      def newIOGeneration(size)
        generation = []
        size.times do
          generation << Neuron.new(self)

        end
        generation

      end

      def completeConnection(firstGeneration, secondGeneration)
        firstGeneration = @generations[firstGeneration]
        secondGeneration = @generations[secondGeneration]

        firstGeneration.each do |neuron|
          secondGeneration.each do |otherNeuron|
            synapse = Synapse.new neuron, otherNeuron, @weights[@weightIndex += 1], self
            neuron.addTerminal synapse
            otherNeuron.addDendrite synapse

          end

        end

      end

  end

  class Neuron
    def initialize(net, bias = nil)
      @net = net
      @dendrites = []
      @terminals = []
      @bias = bias unless bias.nil?

      @output = 0
      @error = 0
      @prevError = 0

    end

    def output=(output)
      @output = output
    end
    def error=(error)
      @error = error
    end

    def terminals
      @terminals
    end
    def output
      @output
    end
    def error
      @error
    end
    def prevError
      @prevError
    end

    def activate
      sigmoid
      @terminals.each do |neuron|
        neuron.transfer

      end

    end

    def train
      sum = 0
      @terminals.each do |synapse|
        sum += synapse.dendrite.error * synapse.weight

      end

      error = derivative * sum
      error = 0 if error.abs <= @net.minimum
      @prevError = @error unless @error == 0
      @error = error

      @terminals.each do |synapse|
        synapse.update

      end

      unless @bias.nil?
        @bias += (1 - @net.decay) * (@net.learning * (@error + @net.momentum * @prevError))
        # @bias += @net.learning * (@error + (@net.momentum * @prevError))
        # @bias += @net.learning * @error
      end

    end

    def sigmoid
      sum = 0
      @dendrites.each do |synapse|
        sum += synapse.input * synapse.weight

      end
      sum += @bias unless @bias.nil?

      @output = 1 / (1 + Math::E**-sum)

    end

    def threshold
      sum = 0
      @dendrites.each do |synapse|
        sum += synapse.input * synapse.weight

      end
      sum += @bias unless@bias.nil?

      @output = 1 / (1 + Math::E**-sum)

      @output =
        if @output < @net.threshold
          0
        else
          1
        end

    end

    def derivative
      @output * (1 - @output)

    end

    def addDendrite(synapse)
      @dendrites << synapse

    end

    def addTerminal(synapse)
      @terminals << synapse

    end

    def printNeuron
      puts @bias
      @terminals.zip(0...@terminals.length).each do |synapse, index|
        print "    "
        print index
        print ": "
        puts synapse.weight

      end

    end

  end

  class Synapse
    def initialize(terminal, dendrite, weight, net)
      @net = net
      @terminal = terminal
      @dendrite = dendrite

      @weight = weight
      @input = 0

    end

    def input
      @input
    end
    def weight
      @weight
    end
    def dendrite
      @dendrite
    end

    def transfer
      @input = @terminal.output

    end

    def update
      @weight = (1 - @net.decay) * (@weight + @net.learning * (@input * (@dendrite.error + @net.momentum * @dendrite.prevError)))
      # @weight += @net.learning * (@input * (@dendrite.error + (@net.momentum * @dendrite.prevError)))
      # @weight += @net.learning * (@input * @dendrite.error)

    end

  end
end







include NeuralNetwork

biases = []
weights = []

(0...2).each do
  biases << rand

end
# biases = [8.736338964129192, 2.8825492034386944]

(0...6).each do
  weights << rand

end

# weights = [-7.265847913093745, -5.569001730108584, -7.268554778059204, -5.5757267828863695, 6.576965166908679, -11.796767425564193]

test = Network.new([2, 2, 1], biases, weights, learning: 0.1, momentum: 0.05, decay: 0, minimum: 0.01)

exitValue = 4

# inputs = [
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ]
# targets = [0, 1, 1, 0]

while true

  exit = 0

  print "Epoch: "
  puts test.epoch

  test.propagate [0, 0]
  exit += test.backpropagate [0]

  test.propagate [0, 1]
  exit += test.backpropagate [1]

  test.propagate [1, 0]
  exit += test.backpropagate [1]

  test.propagate [1, 1]
  exit += test.backpropagate [0]

  break if exit == exitValue

  # print "0, 0: "
  # puts test.propagate [0, 0]
  #
  # print "0, 1: "
  # puts test.propagate [0, 1]
  #
  # print "1, 0: "
  # puts test.propagate [1, 0]
  #
  # print "1, 1: "
  # puts test.propagate [1, 1]
  # puts
  #
  # if test.epoch == 0 then gets end

  if test.epoch % 1000000 == 0
    print "0, 0: "
    puts test.propagate [0, 0]

    print "0, 1: "
    puts test.propagate [0, 1]

    print "1, 0: "
    puts test.propagate [1, 0]

    print "1, 1: "
    puts test.propagate [1, 1]

    test.printNetwork
    gets

    if gets == "break" then break end

  end

  test.epoch = test.epoch + 1

end

puts
puts
puts

print "0, 0: "
puts test.propagate [0, 0]

print "0, 1: "
puts test.propagate [0, 1]

print "1, 0: "
puts test.propagate [1, 0]

print "1, 1: "
puts test.propagate [1, 1]

puts
test.printNetwork


# learning rate - coefficient of every error every step
# momentum - coefficient of previous error added to current error
# weight decay - coefficient of every weights every step
# threshold - a boolean activation function





# weight elimination - coefficient of previous weight subtracted from current weight