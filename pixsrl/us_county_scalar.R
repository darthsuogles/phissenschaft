install.packages(c("sp", "rgeos", "rgdal", "dplyr", "maptools", "leaflet", "scales"))

library(sp)
library(rgeos)
library(rgdal)
library(maptools)
library(dplyr)
library(leaflet)
library(scales)

url = "https://data.cdc.gov/api/views/cjae-szjv/rows.csv?accessType=DOWNLOAD"
dat.epa.water <- read.csv(url, stringsAsFactors = FALSE)

## Colnames tolower
names(dat.epa.water) <- tolower(names(dat.epa.water))
dat.epa.water$countyname <- tolower(dat.epa.water$countyname)

## Wide data set, subset only what we need.
dat.county.cols <- c("reportyear","countyfips","statename", "countyname", "value", "unitname")
dat.county <- subset(dat.epa.water,
                     measureid == "296", 
                     select = dat.county.cols) %>%
    subset(reportyear==2011, select = c("countyfips", "value"))
# Rename columns to make for a clean df merge later.
colnames(dat.county) <- c("GEOID", "airqlty")
# Have to add leading zeos to any FIPS code that's less than 5 digits long to get a good match.
# I'm cheating by using C code. sprintf will work as well.
# >> sprintf("%05d", as.integer(dat.county$GEOID))
dat.county$GEOID <- formatC(dat.county$GEOID, width = 5, format = "d", flag = "0")
### End data prep

# Download county shape file, unzip and put the folder under current directory
# https://www.census.gov/geo/maps-data/data/cbf/cbf_counties.html
us.map <- readOGR(dsn = "cb_2016_us_county_20m", layer = "cb_2016_us_county_20m", stringsAsFactors = FALSE)

# Remove Alaska(2), Hawaii(15), Puerto Rico (72), Guam (66), Virgin Islands (78), American Samoa (60)
#  Mariana Islands (69), Micronesia (64), Marshall Islands (68), Palau (70), Minor Islands (74)
us.map <- us.map[!us.map$STATEFP %in% c("02", "15", "72", "66", "78", "60", "69",
                                        "64", "68", "70", "74"),]
# Make sure other outling islands are removed.
us.map <- us.map[!us.map$STATEFP %in% c("81", "84", "86", "87", "89", "71", "76",
                                        "95", "79"),]
# Merge spatial df with downloade ddata.
leafmap <- merge(us.map, dat.county, by=c("GEOID"))

dat.popup <- paste0("<strong>County: </strong>", 
                    leafmap$NAME, 
                    "<br><strong>Value: </strong>", 
                    leafmap$airqlty)

pal <- colorQuantile("YlOrRd", NULL, n = 20)

## Render final map in leaflet.
leaflet(data = leafmap) %>% addTiles() %>%
    addPolygons(fillColor = ~pal(airqlty), 
                fillOpacity = 0.8, 
                color = "#BDBDC3", 
                weight = 1,
                popup = dat.popup)
