require(tidyverse)
require(rvest)
require(magrittr)
require(plyr)

combineDetails <- function(inV) {
	middle = length(inV)/2
	outV <- c(rep("", middle))
	for (i in 1:middle) {outV[i] = paste(inV[i], inV[middle + i], sep = ": ")}
	return(outV)
}
getDF <- function(URL) {
	pg <- read_html(URL)
	details <- pg %>%
		html_nodes(xpath = "//*[@id='contentN']/table") %>% #found with inspect element's "copy xpath"
		html_text() %>%
		gsub(pattern = "\\\t", replace = "") %>%
		strsplit(split = "\n") %>%
		unlist() %>%
		combineDetails()

	features <- pg %>%
		html_nodes("#contentN > div.rc_stats") %>%
		html_text() %>%
		gsub(pattern = "\\\t", replace = "") %>%
		strsplit(split = "\n") %>%
		unlist() %>%
		magrittr::extract(. != "")

	rcName <- pg %>%
		html_nodes("#contentN > h1") %>%
		html_text()

	rcLocation <- pg %>%
		html_nodes("#contentN > h2 > a") %>%
		html_text()

	rate <- pg %>%
		html_nodes("#contentN > div.ratingblock") %>%
		html_text() %>%
		substr(., gregexpr(., pattern = "Rated: ")[[1]][1], nchar(.)) %>%
		gsub(pattern = "\\\t|\\\n|Rated: |[()]", replace = "") %>%
		str_split(pattern = " ") %>%
		unlist()

	info <- c(details, features) %>%
		str_split(pattern = ": ") %>%
		ldply() %>%
		mutate(name = rcName, park = rcLocation, rating = rate[1], votes = rate[5]) %>%
		select(name, park, rating, votes, cname = V1, detail = V2) %>%
		spread(cname, detail)

	return(info)
}

#use scraping package "rvest" to find all linnks from utimaterollercoaster.com
urc <- read_html("https://www.ultimaterollercoaster.com/coasters/browse/a-to-z") %>%
	html_nodes(".rcyIdx a") %>% #found this css selector using inspect element
	html_attr("href") %>% #get links
	magrittr::extract(. != "#top") %>%
	paste("https://www.ultimaterollercoaster.com", ., sep = '')

details <- lapply(urc, getDF) %>%
	bind_rows() %>%
	rename_all(tolower) %>%
	set_colnames(gsub(gsub(names(.), pattern = "[()]", replacement = ""), pattern = "([[:punct:]])|\\s+", replacement = "_"))

write_delim(details, "scrapedDetails.csv")
save(details, file = "details.RData")
