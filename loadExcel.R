require(tidyverse)

#given data
excel <- suppressWarnings(suppressMessages(read_csv("COMAP_RollerCoasterData_2018.csv"))) %>%
	rename_all(tolower) %>%
	set_colnames(gsub(gsub(names(.), pattern = "[()]", replacement = ""), pattern = "([[:punct:]])|\\s+", replacement = "_")) #remove parentheses, replace punctuation (spaces, slashes, etc.) with underscores

#pipe excel...
excel <- excel %>%
	mutate(height_feet = suppressWarnings(as.numeric(height_feet)), #make the character column numeric
				 duration_sec = 60*60*as.numeric(substr(duration_min_sec, 1, 2)) + 60*as.numeric(substr(duration_min_sec, 4, 5)) + as.numeric(substr(duration_min_sec, 7, 8))) %>% #instead of the duration in hr:mn:sc, sum into seconds
	select(-x20, -status, -duration_min_sec, -inversions_yes_or_no, city = city_region, state = city_state_region, country = country_region, region = geographic_region, year_opened = year_date_opened, inversions = number_of_inversions, height = height_feet, length = length_feet, speed = speed_mph, angle = vertical_angle_degrees) #remove and rename

write_csv(excel, "cleaned_COMAP_data.csv") #write as a new CSV