require(tidyverse)

cc <- read_csv("finalTrainingData.csv") %>%
	rename(designer = manufacturer) %>%
	mutate_if(is.factor, as.character) %>%
	filter(status == "Operating" || status == "Temporarily closed")
urc <- read_csv("exampleDetails.csv") %>%
	mutate_if(is.factor, as.character)
excel <- read_csv("cleaned_COMAP_data.csv") %>%
	mutate_if(is.factor, as.character)

joined <- left_join(cc, urc, by = c("name", "park", "designer", "construction", "opening_year" = "year_opened")) %>%
	mutate(launch = ifelse(is.na(launch.x), launch.y, launch.x),
				 type = ifelse(!is.na(type.y), type.y, type.x),
				 height = ifelse(is.na(height.x), height.y, height.x),
				 inversions = ifelse(is.na(inversions.x), inversions.y, inversions.x),
				 speed = ifelse(is.na(speed.x), speed.y, speed.x),
				 length = ifelse(is.na(length.x), length.y, length.x)) %>%
	select(-launch.x, -launch.y, -type.x, -type.y, -height.x, -height.y, -inversions.x, -inversions.y, -speed.x, -speed.y, -length.x, -length.y, -g_force, -angle, -train_mfg, -drop, -max_passenger, -duration_sec) %>%
	select(-score, score) %>%
	rename(location = park)

write_csv(joined, "joinedTestingData.csv")
