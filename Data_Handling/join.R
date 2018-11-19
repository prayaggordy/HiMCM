require(tidyverse)

cc <- suppressMessages(read_csv("finalTrainingData.csv")) %>%
	rename(designer = manufacturer) %>%
	# mutate_if(is.factor, as.character) %>%
	filter(status == "Operating" || status == "Temporarily closed")
urc <- suppressWarnings(suppressMessages(read_csv("exampleDetails.csv")))# %>%
	# mutate_if(is.factor, as.character)
excel <- suppressMessages(read_csv("cleaned_COMAP_data.csv"))# %>%
	# mutate_if(is.factor, as.character)

northam <- c("United States", "Canada")
europe <- c("Sweden", "Spain", "Poland", "Germany", "Italy", "Netherlands", "United Kingdom", "France", "Denmark", "Finland", "Belgium", "Norway", "Austria")
asia <- c("Japan", "United Arab Emirates", "China", "Singapore")

joined <- left_join(cc, urc, by = c("name", "park", "designer", "construction", "opening_year" = "year_opened")) %>%
	mutate(launch = ifelse(is.na(launch.x), launch.y, launch.x),
				 type = ifelse(!is.na(type.y), type.y, type.x),
				 height = ifelse(is.na(height.x), height.y, height.x),
				 inversions = ifelse(is.na(inversions.x), inversions.y, inversions.x),
				 speed = ifelse(is.na(speed.x), speed.y, speed.x),
				 length = ifelse(is.na(length.x), length.y, length.x)) %>%
	select(-launch.x, -launch.y, -type.x, -type.y, -height.x, -height.y, -inversions.x, -inversions.y, -speed.x, -speed.y, -length.x, -length.y, -g_force, -angle, -train_mfg, -drop, -max_passenger, -duration_sec) %>%
	select(-score, score) %>%
	rename(location = park) %>%
	filter(status != "Definitely closed") %>%
	mutate(country = (ifelse(country %in% northam, "NorthAm", ifelse(country %in% europe, "Europe", "Asia"))), status = as.factor(status), country = as.factor(country), launch = as.factor(launch), type = as.factor(type), name = row_number(), restraint = (ifelse(restraint == "Unknown", NA, ifelse(restraint == "Common lap bar", "Lap bar", ifelse(restraint == "Common shoulder harness", "Shoulder harness", restraint)))), restraint = as.factor(restraint), designer = as.factor(designer), construction = (ifelse(construction == "Boomerang, Family", "Family", ifelse(construction == "Bobsled, Family", "Family", construction))), construction = as.factor(construction)) %>%
	select(-status) %>%
	rename(continent = country) %>%
	mutate_if(is.factor, as.numeric)

noNA <- joined[complete.cases(joined), ]
write_csv(noNA, "noNA.csv")
yesNA <- joined[!complete.cases(joined), ]
write_csv(yesNA, "yesNA.csv")


joined[is.na(joined)] <- 0

# forBryan <- joined %>%
# 	select(name, nameI, continent, countryI, launch, launchI, type, typeI, restraint, restraintI, designer, designerI, construction, constructionI)

# write_csv(forBryan, "textToNum.csv")

# write_csv(joined, "joinedTestingData.csv")
