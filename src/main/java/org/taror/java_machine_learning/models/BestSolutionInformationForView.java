package org.taror.java_machine_learning.models;

import java.util.List;

public class BestSolutionInformationForView {

    private String setOfChromosomes;
    private double error;
    private String weights;
    private List<Double> referenceValues;
    private List<Double> result;

    public BestSolutionInformationForView(String setOfChromosomes, double error, String weights, List<Double> referenceValues, List<Double> result) {
        this.setOfChromosomes = setOfChromosomes;
        this.error = error;
        this.weights = weights;
        this.referenceValues = referenceValues;
        this.result = result;
    }

    public String getSetOfChromosomes() {
        return setOfChromosomes;
    }

    public void setSetOfChromosomes(String setOfChromosomes) {
        this.setOfChromosomes = setOfChromosomes;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public String getWeights() {
        return weights;
    }

    public void setWeights(String weights) {
        this.weights = weights;
    }

    public List<Double> getReferenceValues() {
        return referenceValues;
    }

    public void setReferenceValues(List<Double> referenceValues) {
        this.referenceValues = referenceValues;
    }

    public List<Double> getResult() {
        return result;
    }

    public void setResult(List<Double> result) {
        this.result = result;
    }
}
