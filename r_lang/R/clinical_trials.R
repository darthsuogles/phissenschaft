# https://aact-prod.herokuapp.com/r

source("../import_lib.R")

import.pkgs('RPostgreSQL')

drv <- dbDriver('PostgreSQL')
conn <- dbConnect(drv,
                 dbname="aact",
                 host="aact-prod.cr4nrslb1lw7.us-east-1.rds.amazonaws.com",
                 port=5432,
                 user="aact",
                 password="aact")

SQL <- function(...) { dbGetQuery(conn, paste(..., sep=" ")) }

aact.sample <- SQL("select distinct study_type from studies")

avail.tables <- SQL("SELECT * FROM pg_catalog.pg_tables",
                   "WHERE tablename NOT LIKE 'pg_%' AND tablename NOT LIKE 'sql_%'")

SQL("SELECT column_name, data_type",
    "FROM INFORMATION_SCHEMA.COLUMNS",
    "WHERE table_name = 'studies'")

SQL("SELECT count(*) from studies")

# Fetch data directly from ClinicalTrials.gov
# E.g. open https://clinicaltrials.gov/ct2/results?cond=Blood+Cancer
#      and select the download option to store all data into CSV
import.pkgs("data.table")
studies.df <- fread("blood_cancer_trials.csv")

#as.Date("January 7, 2016", "%B %d, %Y")
studies.df[, date := as.Date(TC$"Last Update Posted", "%B %d, %Y")]

ss <- studies.df[(date > "2016-01-01")][order(date, decreasing = TRUE)]
hist(ss$date, "month", freq = TRUE)
