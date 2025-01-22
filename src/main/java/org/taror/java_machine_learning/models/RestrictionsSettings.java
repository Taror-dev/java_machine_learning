package org.taror.java_machine_learning.models;

import lombok.Data;

@Data
public class RestrictionsSettings {

    private int maxNumberOfInputs;
    private int maxNumberOfNeuronsInTheInputLayer;
    private int maxNumberOfHiddenLayers;
    private int maxNumberOfNeuronsInHiddenLayer;
    private int poolLength;
    private int percentageOfIndividualsForMutation;
    private int percentageOfBestIndividualsInThePool;
}
