require(tidyverse)
require(magrittr)

load("details.RData")

onlyNumber <- function(cl) {
	v <- c(rep(NA, length(cl)))
	for (i in 1:length(cl)) {v[i] = suppressWarnings(as.numeric(strsplit(cl[i], " ")[[1]][1]))}
	return(v)
}
getMaxPassenger <- function(cl) {
	v <- c(rep(0, length(cl)))
	for (i in 1:length(cl)) {v[i] = suppressWarnings(as.numeric(substr(cl[i], str_locate_all(cl[i], " ")[[1]][2], str_locate_all(cl[i], " ")[[1]][3])))}
	return(v)
}

details <- details %>%
	dplyr::rename(construction = track, year_opened = year, speed = top_speed, angle = angle_of_descent) %>%
	mutate_at(c("height", "g_force", "length", "ride_time", "speed"), onlyNumber) %>%
	mutate(angle = as.numeric(str_replace_all(angle, pattern = "°", replacement = "")),
				 max_passenger = getMaxPassenger(trains),
				 year_opened = as.numeric(year_opened)) %>%
	select(-trains)