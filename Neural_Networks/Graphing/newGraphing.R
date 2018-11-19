require(tidyverse)

rsq <- function (x, y) cor(x, y) ^ 2
readAndJoin <- function(backFN, fileIDs) {
	files <- dir(pattern = paste("*", backFN, ".csv", sep = ''))
	df <- files %>%
		map(read_csv, col_names = F) %>%
		bind_cols()
	df <- as_tibble(cbind(nms = names(df), t(df))) %>%
		select(-nms) %>%
		mutate_if(is.character, as.numeric) %>%
		`colnames<-`(c("Expected", "Predicted"))
	fileIDs <- unlist(lapply(fileIDs, rep, times = 49))
	df <- mutate(df, X = c(rep(1:49, nrow(df)/49)), fileID = fileIDs)
	r2s <- c(rep(0, nrow(df)/49))
	for (i in 0:length(r2s)-1) {
		t <- filter(df, fileID == i)
		r2s[i+1] <- rsq(t$Expected, t$Predicted)
	}
	r2s
}

deep <- suppressMessages(readAndJoin("r_deep", c(0, 1, 2, 3)))
shallow <- suppressMessages(readAndJoin("r_shallow", c(0, 1, 2, 3)))
deepE <- suppressMessages(readAndJoin("e_deep", c(0, 1, 2)))
shallowE <- suppressMessages(readAndJoin("e_shallow", c(0, 1, 2)))
