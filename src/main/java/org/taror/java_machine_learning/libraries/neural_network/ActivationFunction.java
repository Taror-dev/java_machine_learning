package org.taror.java_machine_learning.libraries.neural_network;

public class ActivationFunction {

    public static double tanh(double beta, double x) {
        return Math.tanh(beta * x);
    }

    public static double sigmoid(double beta, double x) {
        return 1 / (1 + Math.exp(beta * x * (-1)));
    }

    public static double softmax(double[] arr, double x) {

        double theSumOfTheExponentialsOfAllElements = 0.0;

        for (double v : arr) {
            theSumOfTheExponentialsOfAllElements += Math.exp(v);
        }

        return Math.exp(x) / theSumOfTheExponentialsOfAllElements;
    }
}
