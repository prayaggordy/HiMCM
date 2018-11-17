require(tidyverse)
require(rvest)
require(magrittr)
require(plyr)

URL <- read_file("cleaned.txt") %>%
	str_split("\n") %>%
	unlist() %>%
	paste("https://captaincoaster.com", ., sep = '')

URL <- URL[-length(URL)]

getFeatures <- function(pg, node) {
	features <- pg %>%
		html_node(node) %>%
		html_text() %>%
		gsub(pattern = "  |\\|%|\\\n", replacement = "") %>%
		str_split(pattern = ":") %>%
		unlist() %>%
		na.omit()

	return(features)
}
header.true <- function(df) {
	names(df) <- as.character(unlist(df[1,]))
	df[-1,]
}
fixFeatures <- function(l) {
	a <- header.true(as.data.frame(matrix(l, nrow = 2), stringsAsFactors = F))
	duplicatedCols <- names(a)[duplicated(names(a))]
	if (length(duplicatedCols) != 0) {
		for (i in 1:length(duplicatedCols)) {
			dCols <- which(names(a) == duplicatedCols[i])
			for (j in 2:length(dCols)) {
				colnames(a)[dCols[j]] <- paste(duplicatedCols[i], as.character(j-1), sep = '')
			}
		}
	}

	namesV <- colnames(a)
	dataV <- unname(unlist((a[1,])))
	both <- c(rep("", 2*length(namesV)))
	count = 0
	for (i in seq(1, length(both), by = 2)) {
		count = count + 1
		both[i] = namesV[count]
		both[i + 1] = dataV[count]
	}
	return(both)
}
getInfo <- function(URL) {
	pg <- read_html(URL)
	score <- pg %>%
		html_node("body > div.page-container > div > div.content-wrapper > div.content > div.row > div.col-sm-3 > div:nth-child(1) > div > div.media-body.text-right > h3") %>%
		html_text() %>%
		gsub(pattern = "  |\\\n|%", replacement = "") %>%
		gsub(pattern = ",", replacement = ".") %>%
		str_split(" ") %>%
		unlist() %>%
		nth(3) %>%
		as.numeric()

	height <- pg %>%
		html_node("body > div.page-container > div > div.content-wrapper > div.content > div.row > div.col-sm-3 > div.content-group > div > div:nth-child(1) > button:nth-child(1)") %>%
		html_text() %>%
		gsub(pattern = "  |\\\n|%", replacement = "") %>%
		gsub(pattern = ",", replacement = ".") %>%
		str_split(" ") %>%
		unlist() %>%
		nth(1) %>%
		as.numeric() %>%
		multiply_by(3.2808) #convert to feet

	speed <- pg %>%
		html_node("body > div.page-container > div > div.content-wrapper > div.content > div.row > div.col-sm-3 > div.content-group > div > div:nth-child(1) > button:nth-child(2)") %>%
		html_text() %>%
		gsub(pattern = "  |\\\n|%", replacement = "") %>%
		gsub(pattern = ",", replacement = ".") %>%
		str_split(" ") %>%
		unlist() %>%
		nth(1) %>%
		as.numeric() %>%
		multiply_by(0.621371) #convert to mph

	length <- pg %>%
		html_node("body > div.page-container > div > div.content-wrapper > div.content > div.row > div.col-sm-3 > div.content-group > div > div:nth-child(2) > button:nth-child(1)")  %>%
		html_text() %>%
		gsub(pattern = "  |\\\n|%", replacement = "") %>%
		gsub(pattern = ",", replacement = ".") %>%
		str_split(" ") %>%
		unlist() %>%
		nth(1) %>%
		as.numeric() %>%
		multiply_by(3.2808) #convert to feet

	inversions <- pg %>%
		html_node("body > div.page-container > div > div.content-wrapper > div.content > div.row > div.col-sm-3 > div.content-group > div > div:nth-child(2) > button:nth-child(2)")  %>%
		html_text() %>%
		gsub(pattern = "  |\\\n|%", replacement = "") %>%
		gsub(pattern = ",", replacement = ".") %>%
		str_split(" ") %>%
		unlist() %>%
		nth(1) %>%
		as.numeric()

	name <- pg %>%
		html_node("body > div.page-container > div > div.content-wrapper > div.page-header.page-header-default > div > div.page-title > h1") %>%
		html_text() %>%
		gsub(pattern = "  |\\\n|%", replacement = "") %>%
		str_split(" • ") %>%
		unlist()
	location <- name[2]
	name <- name[1]

	nodes <- paste("body > div.page-container > div > div.content-wrapper > div.content > div.row > div.col-sm-3 > div:nth-child(4) > div.list-group.no-border > div:nth-child(", 1:9, ")", sep = "")
	features <- fixFeatures(unlist(lapply(nodes, getFeatures, pg = pg)))

	return(header.true(as.data.frame(matrix(c("name", name, "score", score, "height", height, "speed", speed, "length", length, "inversions", inversions, features), nrow = 2), stringsAsFactors = F)))
}

detailsCC <- lapply(URL, getInfo) %>%
	bind_rows() %>%
	rename_all(tolower)

# write_delim(detailsCC, "scrapedDetailsCC.csv")
# save(detailsCC, file = "detailsCC.RData")