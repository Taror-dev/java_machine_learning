package org.taror.java_machine_learning.NeuralNetwork;

public class Properties {

    private final String actFun;
    private final int[] layer;
    private final double alpha;
    private final double beta;
    private final double epsilon;
    private final double lambda;

    public Properties(String actFun, int[] layer, double alpha, double beta, double epsilon, double lambda) {
        this.actFun = actFun;
        this.layer = layer;
        this.alpha = alpha;
        this.beta = beta;
        this.epsilon = epsilon;
        this.lambda = lambda;
    }

    public String getActFun() {
        return actFun;
    }

    public double getAlpha() {
        return alpha;
    }

    public double getBeta() {
        return beta;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public double getLambda() {
        return lambda;
    }

    public int[] getLayer() {
        return layer;
    }
}
