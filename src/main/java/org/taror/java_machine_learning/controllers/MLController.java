package org.taror.java_machine_learning.controllers;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.taror.java_machine_learning.entities.TheBestSolution;
import org.taror.java_machine_learning.examples.CheckSolutionXSinXExample;
import org.taror.java_machine_learning.examples.GeneticAlgorithm;
import org.taror.java_machine_learning.models.BestSolutionInformationForView;
import org.taror.java_machine_learning.repositories.SolutionsRepository;
import org.taror.java_machine_learning.repositories.TheBestSolutionRepository;
import org.taror.java_machine_learning.services.ArrayConversions;

import java.util.ArrayList;
import java.util.List;

@Controller
@RequestMapping("/machine_learning")
public class MLController {

    private final TheBestSolutionRepository theBestSolutionRepository;
    private final SolutionsRepository solutionsRepository;
    private final GeneticAlgorithm geneticAlgorithm;

    public MLController(TheBestSolutionRepository theBestSolutionRepository,
                        SolutionsRepository solutionsRepository,
                        GeneticAlgorithm geneticAlgorithm) {
        this.theBestSolutionRepository = theBestSolutionRepository;
        this.solutionsRepository = solutionsRepository;
        this.geneticAlgorithm = geneticAlgorithm;
    }

    @GetMapping("")
    public String mainMenu (Model model) {
        model.addAttribute("isTrainingStarted", geneticAlgorithm.isTrainingStarted());
        return "menu";
    }

    @GetMapping("/start")
    public String start () {

        if (!geneticAlgorithm.isTrainingStarted()) {
            Thread thread = new Thread(() -> {

                try {
                    geneticAlgorithm.run();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
            thread.start();
        }

        return "redirect:";
    }

    @GetMapping("/information_about_the_best_solution")
    public String informationAboutTheBestSolution (Model model) {

        List<TheBestSolution> listBestSolution = theBestSolutionRepository.findByOrderByIdAsc();
        BestSolutionInformationForView bestSolutionInformationForView;

        if (listBestSolution.size() != 0) {

            int[] layers = ArrayConversions.takeLayersFromIndividual(
                    ArrayConversions.stringArrToDoubleArr(listBestSolution.get(0).getSetOfChromosomes()));

            double[] parameters = new double[4];

            for (int i = 0; i < 4; i++) {
                parameters[i] = ArrayConversions.stringArrToDoubleArr(listBestSolution.get(0).getSetOfChromosomes())[3 + i];
            }

            double[][][] weigh = ArrayConversions.weightStringToDoubleArr(layers, listBestSolution.get(0).getWeights());

            CheckSolutionXSinXExample checkSolutionXSinXExample = new CheckSolutionXSinXExample(
                    weigh,
                    "tanh",
                    layers,
                    parameters);
            checkSolutionXSinXExample.run();

            bestSolutionInformationForView = new BestSolutionInformationForView(
                    listBestSolution.get(0).getSetOfChromosomes(),
                    listBestSolution.get(0).getResult(),
                    listBestSolution.get(0).getWeights(),
                    checkSolutionXSinXExample.getReferenceValues(),
                    checkSolutionXSinXExample.getOutResult()
                    );
        } else {

            bestSolutionInformationForView = new BestSolutionInformationForView(
                    String.valueOf(0), 0, String.valueOf(0), new ArrayList<>(), new ArrayList<>());
        }

        model.addAttribute("result", bestSolutionInformationForView);

        return "information_about_the_best_solution";
    }
}
