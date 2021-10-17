package org.taror.java_machine_learning.examples;

import org.taror.java_machine_learning.NeuralNetwork.NeuralNetwork;
import org.taror.java_machine_learning.NeuralNetwork.Properties;

public class AdditionalVerificationOfResults {

    private final Properties properties;
    private final double[][][] weights;
    private boolean notSolution = false;

    public AdditionalVerificationOfResults(double[][][] weights, String actFun, int[] layers, double[] parameters) {

        double alpha = parameters[0];
        double beta = parameters[1];
        double epsilon = parameters[2];
        double lambda = parameters[3];

        this.properties = new Properties(actFun, layers, alpha, beta, epsilon, lambda);
        this.weights = weights;
    }

    public void run() {

        NeuralNetwork neuralNetwork = new NeuralNetwork(properties);
        neuralNetwork.setWeigh(weights);

        double[] in = new double[1];
        double[] out;
        double[] result = new double[40];

        double x = 0;

        for (int i = 0; i < 40; i++) {

            in[0] = x;
            out = neuralNetwork.getResult(in);
            result[i] = out[0];

            x += 0.25;
        }

        int counter = 1;
        for (int i = 1; i < 40; i++) {

            if (result[i] == result[i - 1]) {
                ++counter;
            }
        }

        if (counter == 40) {
            notSolution = true;
        }
    }

    public boolean isNotSolution() {
        return notSolution;
    }
}
