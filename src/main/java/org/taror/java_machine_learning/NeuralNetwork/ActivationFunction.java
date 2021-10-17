package org.taror.java_machine_learning.NeuralNetwork;

public class ActivationFunction {

    public static double tanh(double beta, double x) {
        return Math.tanh(beta * x);
    }

    public static double sigmoid(double beta, double x) {
        return 1 / (Math.exp(beta * x * (-1)));
    }
}
