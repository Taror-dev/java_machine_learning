package org.taror.java_machine_learning.libraries.neural_network;

public class ActivationFunctionDerivative {

    public static double tanh(double beta, double x) {
        return beta * (1 - Math.pow(ActivationFunction.tanh(beta, x), 2));
    }

    public static double sigmoid(double beta, double x) {
        return beta * ActivationFunction.sigmoid(beta, x) * (1 - ActivationFunction.sigmoid(beta, x));
    }
}
