package org.taror.java_machine_learning.examples;

import org.taror.java_machine_learning.NeuralNetwork.NeuralNetwork;
import org.taror.java_machine_learning.NeuralNetwork.Properties;

import java.util.ArrayList;
import java.util.List;

public class CheckSolutionXSinXExample {

    private final Properties properties;
    private final double[][][] weights;

    private List<Double> referenceValues = new ArrayList<>();
    private List<Double> outResult = new ArrayList<>();;

    public CheckSolutionXSinXExample(double[][][] weights, String actFun, int[] layers, double[] parameters) {

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

        double[] input = new double[40];
        double[] answers = new double[40];
        double[] in = new double[1];
        double[] out;

        double x = 0;

        for (int i = 0; i < 40; i++) {

            input[i] = x;
            answers[i] = x * Math.sin(x) / 10;

            in[0] = x;
            out = neuralNetwork.getResult(in);
            referenceValues.add(answers[i]);
            outResult.add(out[0]);

            x += 0.25;
        }
    }

    public List<Double> getReferenceValues() {
        return referenceValues;
    }

    public List<Double> getOutResult() {
        return outResult;
    }
}
