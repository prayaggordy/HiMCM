require(tidyverse)

excel <- read_csv("COMAP_RollerCoasterData_2018.csv") %>% rename_all(tolower)
names(excel) <- gsub(gsub(x = names(excel), pattern = "[()]", replacement = ""), pattern = "([[:punct:]])|\\s+", replacement = "_")

excel <- excel %>%
	mutate(height_feet = as.numeric(height_feet),
				 duration_sec = 60*60*as.numeric(substr(duration_min_sec, 1, 2)) + 60*as.numeric(substr(duration_min_sec, 4, 5)) + as.numeric(substr(duration_min_sec, 7, 8))) %>%
	select(-x20, -status, -duration_min_sec, -inversions_yes_or_no, city = city_region, state = city_state_region, country = country_region, region = geographic_region, year_opened = year_date_opened)

write_csv(excel, "cleaned_COMAP_data.csv")