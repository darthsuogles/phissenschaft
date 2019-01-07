source("../lang/r/import_lib.R")

import.pkgs("devtools")
devtools::install_github("r-lib/devtools")

install_heatmaply <- function() {
    import.pkgs("Rcpp",
                "ggplot2",
                "munsell",
                "htmltools",
                "DBI",
                "assertthat",
                "gridExtra",
                "digest",
                "fpc",
                "TSP",
                "registry",
                "gclus",
                "gplots",
                "RColorBrewer",
                "stringr",
                "labeling",
                "yaml")

    ## make sure you have Rtools installed first! if not, then run:
    ##install.packages('installr'); install.Rtools()
    devtools::install_github("ropensci/plotly") # you will probably benefit from the latest version of plotly
    devtools::install_github('talgalili/heatmaply')
}

install_heatmaply()
