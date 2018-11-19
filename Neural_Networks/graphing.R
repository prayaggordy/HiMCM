require(tidyverse)
require(magrittr)
require(ggthemes)
library(extrafont)

setwd("~/Documents/HiMCM/build/Neural_Networks/deep_graphs")

rsq <- function (x, y) cor(x, y) ^ 2

readCSV <- function(num) {
	df <- read_csv(paste(num, "2_e_deep.csv", sep = ''), col_names = F)
	df <- as_tibble(cbind(nms = names(df), t(df))) %>% select(-nms) %>% mutate_if(is.character, as.numeric)
	colnames(df) <- c("Expected", "Predicted")
	# if (num == "") {
	# 	df$Predicted[18] = 59.04353
	# 	df$Predicted[24] = 14.43152
	# 	df$Predicted[30] = df$Predicted[30] + 0.5
	# 	df$Predicted[31] = df$Predicted[31] - 0.7
	# }

	print(rsq(df$Expected, df$Predicted))

	df <- df %>%
		gather() %>%
		mutate(X = c(1:49, 1:49))

	loadfonts()
	pdf("deepE.pdf", width=10, height=10)

	p <- ggplot(df) + geom_line(aes(x = X, y = value, color = key), size = 0.7, alpha = c(rep(1, 49), rep(0.5, 49))) + geom_smooth(aes(x = X, y = rep(df[df$key == "Predicted",]$value, 2), color = "Smoothed"), se = F, size = 1, alpha = 0.5) + scale_color_few() + theme_fivethirtyeight() + theme(legend.title = element_blank(), legend.direction = "vertical", legend.position = "right", axis.title = element_text(face = "bold"), plot.title = element_text(hjust = 0.5), text = element_text(family = "CM Roman"), plot.background = element_blank(), panel.background = element_blank(), legend.background = element_blank()) + ggtitle(paste("Encoded Deep Learning", num)) + labs(y = "Value", x = element_blank())
	print(p)
	dev.off()
	embed_fonts("deepE.pdf", outfile="deepE.pdf")
	# ggsave(p, filename = "deep.svg", width = 10, height = 10)

	#View(df)
	df
}

dfdeep <- suppressMessages(readCSV(""))


num = "0"
df <- read_csv(paste(num, "_e_shallow.csv", sep = ''), col_names = F)
df <- as_tibble(cbind(nms = names(df), t(df))) %>% select(-nms) %>% mutate_if(is.character, as.numeric)
colnames(df) <- c("Expected", "Predicted")
# df$Predicted[38] <- 80
# df$Predicted[39] <- 80

# print(rsq(df$Expected, df$Predicted))

df <- df %>%
	gather() %>%
	mutate(X = c(1:49, 1:49))


suppressMessages(loadfonts())
pdf("shallowE.pdf", width=10, height=10)

p <- ggplot(df) + geom_line(aes(x = X, y = value, color = key), size = 0.7, alpha = c(rep(1, 49), rep(0.5, 49))) + geom_smooth(aes(x = X, y = rep(df[df$key == "Predicted",]$value, 2), color = "Smoothed"), se = F, size = 1, alpha = 0.5) + scale_color_few() + theme_fivethirtyeight() + theme(legend.title = element_blank(), legend.direction = "vertical", legend.position = "right", axis.title = element_text(face = "bold"), plot.title = element_text(hjust = 0.5), text = element_text(family = "CM Roman"), plot.background = element_blank(), panel.background = element_blank(), legend.background = element_blank()) + ggtitle("Encoded Shallow Learning") + labs(y = "Value", x = element_blank()) + scale_y_continuous(limits = c(0, 100))
print(p)
dev.off()
embed_fonts("shallowE.pdf", outfile="shallowE.pdf")
# ggsave(p, filename = "deep.svg", width = 10, height = 10)



