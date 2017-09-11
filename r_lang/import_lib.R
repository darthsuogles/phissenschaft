
import.pkgs <- function (...) {
    df.pkgs.loaded <- data.frame(installed.packages())
    pkgs.loaded <- levels(df.pkgs.loaded$Package)
    pkgs.required <- list(...)
    pkgs.todo <- setdiff(pkgs.required, pkgs.loaded)
    for (pkg in pkgs.todo) {
        install.packages(pkg, repos = c("https://cloud.r-project.org"))
    }
    for (pkg in pkgs.required) {
        suppressWarnings(require(pkg, character.only = TRUE))
    }
}

import.pkgs('plyr', 'glmnet', 'caret', 'ggplot2', 'prophet', 'forecast')
