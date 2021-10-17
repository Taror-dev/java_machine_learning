package org.taror.java_machine_learning.examples;

import org.springframework.stereotype.Component;
import org.taror.java_machine_learning.GeneticAlgorithm.Crossbreeding;
import org.taror.java_machine_learning.GeneticAlgorithm.GenerateIndividual;
import org.taror.java_machine_learning.GeneticAlgorithm.Mutation;
import org.taror.java_machine_learning.GeneticAlgorithm.model.Individual;
import org.taror.java_machine_learning.entities.Solutions;
import org.taror.java_machine_learning.entities.TheBestSolution;
import org.taror.java_machine_learning.repositories.SolutionsRepository;
import org.taror.java_machine_learning.repositories.TheBestSolutionRepository;
import org.taror.java_machine_learning.services.ArrayConversions;
import org.taror.java_machine_learning.services.RandomNumber;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

@Component
public class GeneticAlgorithm {

    private final TheBestSolutionRepository theBestSolutionRepository;
    private final SolutionsRepository solutionsRepository;

    private final int maxInputLayersCount = 1;
    private final int maxHiddenLayersCount = 10;
    private final int maximumNumberOfNeuronsInTheHiddenLayer = 10;

    private final Map<Integer, Individual> resultAndIndividuals = new HashMap<>();

    private double bestOfAllResult = 1.0;

    private boolean trainingStarted = false;

    public GeneticAlgorithm(TheBestSolutionRepository theBestSolutionRepository, SolutionsRepository solutionsRepository) {
        this.theBestSolutionRepository = theBestSolutionRepository;
        this.solutionsRepository = solutionsRepository;
    }

    public void run() throws InterruptedException {

        trainingStarted = true;

        List<TheBestSolution> listBestSolution = theBestSolutionRepository.findByOrderByIdAsc();

        if (listBestSolution.size() != 0) {
            TheBestSolution theBestSolution = listBestSolution.get(0);

            bestOfAllResult = theBestSolution.getResult();
        }


        int lengthPool = 100;
        int percentageOfIndividualsForMutation = 50;
        int percentageOfBestIndividualsInThePool = 40;

        int counter;

        List<Individual> buffer = new ArrayList<>();
        List<Individual> solutionPool = new ArrayList<>();
        List<Individual> solutionPoolDescendants = new ArrayList<>();

        int id = 0;

        while (true) {

            List<Solutions> solutionsInDb = solutionsRepository.findByOrderByIdAsc();

            if (solutionsInDb.size() == 0) {

                //Генерация особей
                int needToGenerateIndividuals = lengthPool - solutionPool.size();

                for (int i = 0; i < needToGenerateIndividuals; i++) {
                    solutionPool.add(new Individual(++id,
                            GenerateIndividual.randomIndividual(maxInputLayersCount, maxHiddenLayersCount,
                                    maximumNumberOfNeuronsInTheHiddenLayer)));
                }

                //Выбор особей для мутации
                int numberOfMutatedIndividuals = (int) Math.ceil(lengthPool * percentageOfIndividualsForMutation / 100.0);
                List<Integer> listOfMutatedIndividuals = Mutation.RandomSelectionOfIndividualsForMutation(numberOfMutatedIndividuals, lengthPool);

                //Мутация выбранных особей
                for (int numbersOfMutatedIndividual : listOfMutatedIndividuals) {
                    buffer.add(Mutation.changingTheRandomChromosomeToTheMinimumValue(
                            ++id,
                            solutionPool.get(numbersOfMutatedIndividual),
                            maxInputLayersCount,
                            maxHiddenLayersCount,
                            maximumNumberOfNeuronsInTheHiddenLayer));
                }

                solutionPool.addAll(buffer);
                buffer.clear();

                //Выбрать особей для скрещивания (случайно)
                List<Individual> selectedPairsForCrossing = new ArrayList<>();
                List<Individual> parentsPool = new ArrayList<>();

                while (solutionPool.size() != 0) {

                    if (solutionPool.size() == 1) {
                        parentsPool.add(solutionPool.get(0));
                        solutionPool.remove(0);
                        break;
                    }

                    for (int i = 0; i < 2; i++) {
                        int individualNumber = (int) Math.round(RandomNumber.randomInRange(0, solutionPool.size() - 1));
                        selectedPairsForCrossing.add(solutionPool.get(individualNumber));
                        parentsPool.add(solutionPool.get(individualNumber));
                        solutionPool.remove(individualNumber);
                    }

                    //Скрещивание
                    solutionPoolDescendants.add(Crossbreeding.oneDescendantRandomGenes(++id, selectedPairsForCrossing));
                    selectedPairsForCrossing.clear();
                }

                //Добалять родителей в пул тоже
                solutionPoolDescendants.addAll(parentsPool);

                for (Individual individual : solutionPoolDescendants) {
                    solutionsRepository.save(new Solutions(individual.getId(), individual.getAge(),
                            Arrays.toString(individual.getSetOfChromosomes()), individual.getResult()));
                }

            } else {
                for (Solutions solutions : solutionsInDb) {

                    Individual individual = new Individual(
                            solutions.getId(),
                            ArrayConversions.stringArrToDoubleArr(solutions.getSetOfChromosomes()));

                    individual.setAge(solutions.getAge());
                    individual.setResult(solutions.getResult());

                    solutionPoolDescendants.add(individual);
                }
            }

            //Оценка
            counter = 0;

            int processors = Runtime.getRuntime().availableProcessors();
            ExecutorService threadPool = Executors.newFixedThreadPool(processors - 2);

            for (Individual individual : solutionPoolDescendants) {

                ++counter;

                if (individual.getResult() == 0.0) {

                    int finalCounter = counter;
                    threadPool.submit(
                            new Thread(() -> {
                                runNN(finalCounter, individual);
                            })
                    );

                } else {
                    resultAndIndividuals.put(counter, individual);
                }
            }

            threadPool.shutdown();
            threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);

            //Отбор лучших
            int numberOfBestIndividualsInThePool = (int) Math.ceil(lengthPool * percentageOfBestIndividualsInThePool / 100.0);
            int number = 0;
            double bestResult;

            Individual bestIndividual = new Individual(0, new double[7]);

            for (int i = 0; i < numberOfBestIndividualsInThePool; i++) {

                bestResult = 1.0;

                for (Map.Entry<Integer, Individual> entry : resultAndIndividuals.entrySet()) {

                    if (entry.getValue().getResult() < bestResult) {
                        number = entry.getKey();
                        bestResult = entry.getValue().getResult();
                        bestIndividual = entry.getValue();
                    }
                }

                if (bestIndividual.getAge() < 5) {
                    bestIndividual.setAge(bestIndividual.getAge() + 1);
                    solutionPool.add(bestIndividual);
                }

                resultAndIndividuals.remove(number);
            }

            solutionPoolDescendants.clear();
            solutionsRepository.deleteAll();

            if (solutionPool.get(0).getResult() < 0.003) {//------------------------------------------------------------
                break;//------------------------------------------------------------------------------------------------
            }//---------------------------------------------------------------------------------------------------------
        }

        trainingStarted = false;
    }

    private void runNN(int counter, Individual individual) {

        int[] layers = ArrayConversions.takeLayersFromIndividual(individual.getSetOfChromosomes());

        double[] parameters = new double[4];

        for (int i = 0; i < 4; i++) {
            parameters[i] = individual.getSetOfChromosomes()[3 + i];
        }

        NeuralNetworkXSinX neuralNetworkXSinX = new NeuralNetworkXSinX(40, 51200, 4, "tanh", layers, parameters);
        neuralNetworkXSinX.run();

        AdditionalVerificationOfResults additionalVerificationOfResults = new AdditionalVerificationOfResults(
                neuralNetworkXSinX.getWeigh(), "tanh", layers, parameters);
        additionalVerificationOfResults.run();

        double result = neuralNetworkXSinX.getError();

        individual.setResult(result);

        synchronized (this) {

            if (result < bestOfAllResult && !additionalVerificationOfResults.isNotSolution()) {

                bestOfAllResult = result;

                theBestSolutionRepository.deleteAll();
                theBestSolutionRepository.save(new TheBestSolution(individual.getId(), individual.getResult(),
                        individual.getSetOfChromosomes(), neuralNetworkXSinX.getWeigh()));
            }
        }

        resultAndIndividuals.put(counter, individual);
        solutionsRepository.save(new Solutions(individual.getId(), individual.getAge(),
                Arrays.toString(individual.getSetOfChromosomes()), result));
    }

    public boolean isTrainingStarted() {
        return trainingStarted;
    }
}
