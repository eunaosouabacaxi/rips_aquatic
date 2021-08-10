library("party")
library("arules")
library("rpart")
library("tidyverse")

features <- read.csv("/u/project/cratsch/tescala/output_for_mob/top_feats_1_5_early_50.csv")

x <- c()
z <- c()
for (i in 1:34) {
  str <- paste("x", i, sep = "")
  x <- append(x, str)
}
for (i in 1:12) {
  str <- paste("z", i, sep = "")
  z <- append(z, str)
}
x_ctr <- c()
z_ctr <- c()
for (i in 1:34) {
  x_list <- nrow(which(features == x[i], arr.ind = TRUE))
  x_ctr <- append(x_ctr, x_list)
}
for (i in 1:12) {
  z_list <- nrow(which(features == z[i], arr.ind = TRUE))
  z_ctr <- append(z_ctr, z_list)
}
print(x_ctr)
print(z_ctr)

max_x_list <- c()
for (i in 1:5) {
  max_x_list <- append(max_x_list, which.max(x_ctr))
  x_ctr[which.max(x_ctr)] <- 0
}
print(x_ctr)
print(max_x_list)

x_top <- c()
for (i in 1:5) {
  x_top <- append(x_top, x[max_x_list[i]])
}
print(x_top)

if (!is.na(match(x_top[1], x))) {
  f <- match(x_top[1], x))
if (f == 1) a <- df$x1
if (f == 2) a <- df$x2
if (f == 3) a <- df$x3
if (f == 4) a <- df$x4
if (f == 5) a <- df$x5
if (f == 6) a <- df$x6
if (f == 7) a <- df$x7
if (f == 8) a <- df$x8
if (f == 9) a <- df$x9
if (f == 10) a <- df$x10
if (f == 11) a <- df$x11
if (f == 12) a <- df$x12
if (f == 13) a <- df$x13
if (f == 14) a <- df$x14
if (f == 15) a <- df$x15
if (f == 16) a <- df$x16
if (f == 17) a <- df$x17
if (f == 18) a <- df$x18
if (f == 19) a <- df$x19
if (f == 20) a <- df$x20
if (f == 21) a <- df$x21
if (f == 22) a <- df$x22
if (f == 23) a <- df$x23
if (f == 24) a <- df$x24
if (f == 25) a <- df$x25
if (f == 26) a <- df$x26
if (f == 27) a <- df$x27
if (f == 28) a <- df$x28
if (f == 29) a <- df$x29
if (f == 30) a <- df$x30
if (f == 31) a <- df$x31
if (f == 32) a <- df$x32
if (f == 33) a <- df$x33
if (f == 34) a <- df$x34
}

if (!is.na(match(x_top[2], x))) {
  f <- match(x_top[2], x))
if (f == 1) b <- df$x1
if (f == 2) b <- df$x2
if (f == 3) b <- df$x3
if (f == 4) b <- df$x4
if (f == 5) b <- df$x5
if (f == 6) b <- df$x6
if (f == 7) b <- df$x7
if (f == 8) b <- df$x8
if (f == 9) b <- df$x9
if (f == 10) b <- df$x10
if (f == 11) b <- df$x11
if (f == 12) b <- df$x12
if (f == 13) b <- df$x13
if (f == 14) b <- df$x14
if (f == 15) b <- df$x15
if (f == 16) b <- df$x16
if (f == 17) b <- df$x17
if (f == 18) b <- df$x18
if (f == 19) b <- df$x19
if (f == 20) b <- df$x20
if (f == 21) b <- df$x21
if (f == 22) b <- df$x22
if (f == 23) b <- df$x23
if (f == 24) b <- df$x24
if (f == 25) b <- df$x25
if (f == 26) b <- df$x26
if (f == 27) b <- df$x27
if (f == 28) b <- df$x28
if (f == 29) b <- df$x29
if (f == 30) b <- df$x30
if (f == 31) b <- df$x31
if (f == 32) b <- df$x32
if (f == 33) b <- df$x33
if (f == 34) b <- df$x34
}

if (!is.na(match(x_top[3], x))) {
  f <- match(x_top[3], x))
if (f == 1) c <- df$x1
if (f == 2) c <- df$x2
if (f == 3) c <- df$x3
if (f == 4) c <- df$x4
if (f == 5) c <- df$x5
if (f == 6) c <- df$x6
if (f == 7) c <- df$x7
if (f == 8) c <- df$x8
if (f == 9) c <- df$x9
if (f == 10) c <- df$x10
if (f == 11) c <- df$x11
if (f == 12) c <- df$x12
if (f == 13) c <- df$x13
if (f == 14) c <- df$x14
if (f == 15) c <- df$x15
if (f == 16) c <- df$x16
if (f == 17) c <- df$x17
if (f == 18) c <- df$x18
if (f == 19) c <- df$x19
if (f == 20) c <- df$x20
if (f == 21) c <- df$x21
if (f == 22) c <- df$x22
if (f == 23) c <- df$x23
if (f == 24) c <- df$x24
if (f == 25) c <- df$x25
if (f == 26) c <- df$x26
if (f == 27) c <- df$x27
if (f == 28) c <- df$x28
if (f == 29) c <- df$x29
if (f == 30) c <- df$x30
if (f == 31) c <- df$x31
if (f == 32) c <- df$x32
if (f == 33) c <- df$x33
if (f == 34) c <- df$x34
}

if (!is.na(match(x_top[4], x))) {
  f <- match(x_top[4], x))
if (f == 1) d <- df$x1
if (f == 2) d <- df$x2
if (f == 3) d <- df$x3
if (f == 4) d <- df$x4
if (f == 5) d <- df$x5
if (f == 6) d <- df$x6
if (f == 7) d <- df$x7
if (f == 8) d <- df$x8
if (f == 9) d <- df$x9
if (f == 10) d <- df$x10
if (f == 11) d <- df$x11
if (f == 12) d <- df$x12
if (f == 13) d <- df$x13
if (f == 14) d <- df$x14
if (f == 15) d <- df$x15
if (f == 16) d <- df$x16
if (f == 17) d <- df$x17
if (f == 18) d <- df$x18
if (f == 19) d <- df$x19
if (f == 20) d <- df$x20
if (f == 21) d <- df$x21
if (f == 22) d <- df$x22
if (f == 23) d <- df$x23
if (f == 24) d <- df$x24
if (f == 25) d <- df$x25
if (f == 26) d <- df$x26
if (f == 27) d <- df$x27
if (f == 28) d <- df$x28
if (f == 29) d <- df$x29
if (f == 30) d <- df$x30
if (f == 31) d <- df$x31
if (f == 32) d <- df$x32
if (f == 33) d <- df$x33
if (f == 34) d <- df$x34
}

if (!is.na(match(x_top[5], x))) {
  f <- match(x_top[5], x))
if (f == 1) e <- df$x1
if (f == 2) e <- df$x2
if (f == 3) e <- df$x3
if (f == 4) e <- df$x4
if (f == 5) e <- df$x5
if (f == 6) e <- df$x6
if (f == 7) e <- df$x7
if (f == 8) e <- df$x8
if (f == 9) e <- df$x9
if (f == 10) e <- df$x10
if (f == 11) e <- df$x11
if (f == 12) e <- df$x12
if (f == 13) e <- df$x13
if (f == 14) e <- df$x14
if (f == 15) e <- df$x15
if (f == 16) e <- df$x16
if (f == 17) e <- df$x17
if (f == 18) e <- df$x18
if (f == 19) e <- df$x19
if (f == 20) e <- df$x20
if (f == 21) e <- df$x21
if (f == 22) e <- df$x22
if (f == 23) e <- df$x23
if (f == 24) e <- df$x24
if (f == 25) e <- df$x25
if (f == 26) e <- df$x26
if (f == 27) e <- df$x27
if (f == 28) e <- df$x28
if (f == 29) e <- df$x29
if (f == 30) e <- df$x30
if (f == 31) e <- df$x31
if (f == 32) e <- df$x32
if (f == 33) e <- df$x33
if (f == 34) e <- df$x34
}


