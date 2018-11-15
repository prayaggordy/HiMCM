load("detailsCC.RData")
require(tidyverse)

detailsCC <- detailsCC %>%
	dplyr::rename(construction = type, type = type1) %>%
	set_colnames(gsub(gsub(names(.), pattern = "[()]", replacement = ""), pattern = "([[:punct:]])|\\s+", replacement = "_")) %>%
	mutate(vr = ifelse(is.na(vr), F, T),
				 opening_year = ifelse(substring(opening_date, nchar(opening_date)-2, nchar(opening_date)-2) == "/", ifelse(as.integer(substring(opening_date, nchar(opening_date)-1, nchar(opening_date)))<=18, paste(20, substring(opening_date, nchar(opening_date)-1, nchar(opening_date)), sep = ''), paste(19, substring(opening_date, nchar(opening_date)-1, nchar(opening_date)), sep = '')), opening_date)) %>%
	select(name, park, country, score, status, manufacturer, construction, launch, restraint, type, opening_year, height, speed, length, inversions)

write_csv(details, "exampleDetails.csv")