package org.taror.java_machine_learning.services;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.taror.java_machine_learning.libraries.neural_network.NeuralNetwork;
import org.taror.java_machine_learning.libraries.neural_network.models.ActFun;
import org.taror.java_machine_learning.libraries.neural_network.models.LossFun;
import org.taror.java_machine_learning.models.Individual;

import java.util.ArrayList;
import java.util.List;

@Data
public class NNSinX {

    private long maxNumberOfIterations = 52100;
    private int numberOfInputs;
    private int[] layers;

    private ActFun actFun;
    private LossFun lossFun;
    private double alpha;             //form 0 to 1
    private double beta;              //form 0 to 1
    private double epsilon;           //form 0 to 1
    private double lambda;            //form 0 to 0,9

    private Double error;
    private List<Double> inputs = new ArrayList<>();
    private List<Double> answers = new ArrayList<>();
    private List<Double> results = new ArrayList<>();

    public NNSinX(Individual individual, ActFun actFun, LossFun lossFun) {

        this.numberOfInputs = (int) individual.getSetOfChromosomes()[0];

        layers = new int[(int) individual.getSetOfChromosomes()[2] + 2];
        layers[0] = 1;

        for (int i = 1; i <= layers.length - 1; i++) {
            layers[i] = (int) individual.getSetOfChromosomes()[3];
        }

        layers[(int) individual.getSetOfChromosomes()[2] + 1] = 1;

        this.actFun = actFun;
        this.lossFun = lossFun;
        this.alpha = individual.getSetOfChromosomes()[4];
        this.beta = individual.getSetOfChromosomes()[5];
        this.epsilon = individual.getSetOfChromosomes()[6];
        this.lambda = individual.getSetOfChromosomes()[7];
    }

    public void run() {

        NeuralNetwork neuralNetwork = new NeuralNetwork(actFun, lossFun, layers, alpha, beta, epsilon, lambda);
        neuralNetwork.initializeWeightsWithRandomValues(1);

        double[] input = new double[numberOfInputs];
        double[] answer = new double[numberOfInputs];

        double x = 0;

        for (int i = 0; i < numberOfInputs; i++) {

            input[i] = x;
            answer[i] = x * Math.sin(x*10);
            x = x + 0.025;

            inputs.add(input[i]);
            answers.add(answer[i]);
        }

        int counter = 0;

        while (true) {

            long end = System.currentTimeMillis() + 200;
            while (System.currentTimeMillis() < end) {

                for (int i = 0; i < numberOfInputs; i++) {

                    neuralNetwork.updateValueInNeurons(new double[]{input[i]});
                    neuralNetwork.backPropagationOfError(neuralNetwork.getResult(), new double[]{answer[i]});
                    neuralNetwork.gradientCalculation();
                    neuralNetwork.weighUpdate();
                }

                counter++;

                error = 0.0;

                for (int j = 0; j < numberOfInputs; j++) {
                    neuralNetwork.updateValueInNeurons(new double[]{input[j]});
                    error += answer[j] - neuralNetwork.getResult()[0];
                }

                error = Math.abs(error / numberOfInputs);

                if (counter > maxNumberOfIterations) {
                    break;
                }
            }

            results.clear();

            for (int j = 0; j < numberOfInputs; j++) {

                neuralNetwork.updateValueInNeurons(new double[]{input[j]});
                results.add(neuralNetwork.getResult()[0]);
            }

            if (counter > maxNumberOfIterations) {
                break;
            }
        }
    }
}
