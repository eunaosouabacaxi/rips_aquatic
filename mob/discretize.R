df <- readRDS("features_jan_morning_2015.rds")
head(df)

df$Unnamed..0_x <- NULL
df$Unnamed..0_y <- NULL

# two-way inclusive
for(i in 5:38) {
  # renaming predictors with X_i
  names(df)[i] <- paste("X_", toString(i - 4), sep = "")
}

print(names(df)[38])

names(df)[39] <- "Z_1"
names(df)[43] <- "Z_2"
names(df)[47] <- "Z_3"
names(df)[51] <- "Z_4"
names(df)[55] <- "Z_5"
names(df)[59] <- "Z_6"
names(df)[63] <- "Z_7"
names(df)[67] <- "Z_8"
names(df)[71] <- "Z_9"
names(df)[75] <- "Z_10"
names(df)[79] <- "Z_11"
names(df)[83] <- "Z_12"
names(df)[87] <- "Z_13"
names(df)[88] <- "Z_14"
names(df)[89] <- "Z_15"

head(df)
print(names(df)[90])

names(df)[90] <- "Y_2"
names(df)[91] <- "Y_1"

library("arules")
disc <- discretizeDF(df[,5:89], default = list(method = "interval", breaks = 10,
                                       labels = 1:10))

df_1 <- df[,1:4]
head(df_1)

df_2 <- df[,90:95]
head(df_2)

df_merged <- merge(disc, df_2)
rm(df)

df_1.saveRDS()
