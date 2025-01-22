package org.taror.java_machine_learning.configuration;

import lombok.Data;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Configuration
@Data
public class AppConfig {

    @Value("${app.number-of-processors}")
    private int numberOfProcessors;
}
