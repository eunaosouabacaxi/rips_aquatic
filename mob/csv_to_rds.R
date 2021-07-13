for (i in 1:50) {
  fin = paste("processed_2012_", toString(i), sep = "")
  fin = paste(fin, ".csv", sep = "")
  df_2012 <- read.csv(file = fin)
  fout = paste("combined_2012_", toString(i), sep = "")
  fout = paste(fout, ".rds", sep = "")
  saveRDS(df_2012, file = fout)
}

# read in preprocessed combined files
#df_2012 <- read.csv(file = 'processed_2012_1.csv')
# Save an object to a file
#saveRDS(df_2012, file = "combined_2012_1.rds")
# Restore the object
#readRDS(file = "my_data.rds")