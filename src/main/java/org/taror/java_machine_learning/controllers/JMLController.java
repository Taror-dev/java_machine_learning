package org.taror.java_machine_learning.controllers;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.taror.java_machine_learning.services.FindingSolutionUsingGA;

@Slf4j
@Controller
@RequestMapping("/java_machine_learning")
@RequiredArgsConstructor
public class JMLController {

    private final FindingSolutionUsingGA solution;

    @GetMapping("")
    public String mainMenu (Model model) {
        model.addAttribute("isTrainingStarted", solution.isTrainingStarted());
        return "menu";
    }

    @GetMapping("/start")
    public String start () {

        Thread thread = new Thread(() -> {
            try {
                solution.run();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        });

        thread.start();
        log.info("Start");

        return "redirect:/java_machine_learning";
    }

    @GetMapping("/stop")
    public String stop () {

        solution.setTrainingStarted(false);
        log.info("Stop");

        return "redirect:/java_machine_learning";
    }

    @GetMapping("/information_about_the_best_solution")
    public String informationAboutTheBestSolution (Model model) {

        model.addAttribute("solution", solution.getTheBestIndividual());

        return "information_about_the_best_solution";
    }
}
